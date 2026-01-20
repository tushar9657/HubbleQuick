import re, csv, html, time, shutil, threading, random, io, gzip, zipfile, os
from pathlib import Path
from datetime import datetime, timedelta, time as dtime
from collections import defaultdict
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, unquote

import requests
import xml.etree.ElementTree as ET
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    IST = None

# -----------------------------
# Settings
# -----------------------------
RSS_URL = "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml"

# Workdir inside GitHub workspace (override via WORKDIR env var if needed)
_DEFAULT_WORKDIR = Path(os.environ.get("GITHUB_WORKSPACE", ".")) / "nse_daily_runs"
WORKDIR = Path(os.environ.get("WORKDIR", str(_DEFAULT_WORKDIR)))
WORKDIR.mkdir(parents=True, exist_ok=True)

RSS_FETCH_MAX_RETRIES = 8
RSS_FETCH_TIMEOUT = (6, 30)  # (connect, read)

# Pipeline Settings
CLUSTER_SECONDS = 60  # keep original intent; 1 sec is too strict/noisy

EXISTS_BATCH_SIZE = 25
EXISTS_MAX_WORKERS = 6
EXISTS_BATCH_PAUSE_S = 0.6
EXISTS_TIMEOUT = (6, 20)   # (connect, read)

PROCESS_MAX_WORKERS = 4
FETCH_TIMEOUT = (8, 60)    # (connect, read)
MAX_RETRIES = 3

BACKOFF_BASE_S = 0.6
MAX_BACKOFF_S = 60
JITTER_FRAC = 0.25

PDF_TEXT_MAX_PAGES = 12
PDF_TEXT_RETRY_MAX_PAGES = 24

OCR_MAX_PAGES = 6
OCR_SCALE = 2.0
OCR_TEXTCHAR_THRESHOLD = 2500

OCR_RETRY_MAX_PAGES = 12
OCR_RETRY_SCALE = 2.5
OCR_RETRY_TESS_CONFIG = "--oem 1 --psm 6"

# XML retry for Not Extracted
XML_RETRY_ATTEMPTS = 2
XML_RETRY_WORKERS = 2

CONTENT_CAP = 100_000

NSE_HOME = "https://www.nseindia.com/"
NSE_REFERER = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"

OCR_ENABLED = shutil.which("tesseract") is not None

# -----------------------------
# Time helpers (IST)
# -----------------------------
def now_ist() -> datetime:
    if IST is not None:
        return datetime.now(IST).replace(tzinfo=None)
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).replace(tzinfo=None)

def run_label_date_ist() -> str:
    """
    Label outputs by IST date. If GitHub starts a few minutes late (after midnight IST),
    label as previous day to match the intended 23:59 IST run.
    """
    n = now_ist()
    if n.time() < dtime(0, 30):  # small jitter window
        n = n - timedelta(days=1)
    return n.date().isoformat()

# -----------------------------
# Thread-local HTTP session
# -----------------------------
_tls = threading.local()
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "DNT": "1",
    "Referer": NSE_REFERER,
}

def get_session() -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update(DEFAULT_HEADERS)
        _tls.session = s
        _tls.nse_warmed = False
    return s

def backoff(attempt: int):
    base = min(MAX_BACKOFF_S, BACKOFF_BASE_S * (2 ** (attempt - 1)))
    time.sleep(base * (1.0 + random.random() * JITTER_FRAC))

def _hostname(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().strip()
    except Exception:
        return ""

def _needs_nse_warmup(url: str) -> bool:
    h = _hostname(url)
    return h.endswith("nseindia.com") and not h.startswith("nsearchives.")

def ensure_nse_warmup(s: requests.Session):
    if getattr(_tls, "nse_warmed", False):
        return
    try:
        s.get(NSE_HOME, timeout=EXISTS_TIMEOUT)
        s.get(NSE_REFERER, timeout=EXISTS_TIMEOUT)
    except Exception:
        pass
    _tls.nse_warmed = True

# -----------------------------
# URL helpers
# -----------------------------
def url_suffix(url: str) -> str:
    try:
        p = unquote(urlparse(url).path or "")
        return Path(p).suffix.lower()
    except Exception:
        return ""

def is_pdf_url(url: str) -> bool:
    return url_suffix(url) == ".pdf"

def is_xml_url(url: str) -> bool:
    u = (url or "").lower()
    return url_suffix(url) == ".xml" or "/xbrl/" in u

# -----------------------------
# Exists check (HEAD → Range GET → probe GET)
# -----------------------------
def exists_check(url: str) -> bool:
    url = (url or "").strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return False

    s = get_session()
    if _needs_nse_warmup(url):
        ensure_nse_warmup(s)

    retryable = {403, 405, 429, 500, 502, 503, 504}
    not_found = {404, 410}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = s.head(url, allow_redirects=True, timeout=EXISTS_TIMEOUT)
            status = r.status_code
            r.close()
            if status in not_found:
                return False
            if status < 400:
                return True
            if status in retryable or status is None:
                raise RuntimeError(f"HEAD_STATUS_{status}")
        except Exception:
            pass

        try:
            r = s.get(
                url,
                headers={"Range": "bytes=0-1023"},
                stream=True,
                allow_redirects=True,
                timeout=EXISTS_TIMEOUT,
            )
            status = r.status_code
            r.close()
            if status in not_found:
                return False
            if status == 416:
                return True
            if status < 400:
                return True
            if status in retryable or status is None:
                raise RuntimeError(f"RANGE_STATUS_{status}")
        except Exception:
            pass

        try:
            r = s.get(url, stream=True, allow_redirects=True, timeout=EXISTS_TIMEOUT)
            status = r.status_code
            if status in not_found:
                r.close()
                return False
            if status < 400:
                try:
                    for _ in r.iter_content(chunk_size=1):
                        break
                finally:
                    r.close()
                return True
            r.close()
        except Exception:
            backoff(attempt)

    return True  # optimistic

# -----------------------------
# Fetch bytes
# -----------------------------
def fetch_bytes(url: str) -> tuple[bool, bytes]:
    url = (url or "").strip()
    if not url:
        return False, b""

    s = get_session()
    if _needs_nse_warmup(url):
        ensure_nse_warmup(s)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = s.get(url, allow_redirects=True, timeout=FETCH_TIMEOUT)
            status = r.status_code
            b = r.content
            r.close()
            if status in (404, 410):
                return False, b""
            if status >= 400 or not b:
                backoff(attempt)
                continue
            return True, b
        except Exception:
            backoff(attempt)
    return False, b""

# -----------------------------
# RSS fetch
# -----------------------------
def fetch_rss_xml() -> str:
    s = get_session()
    try:
        ensure_nse_warmup(s)
    except Exception:
        pass

    last_err = None
    for attempt in range(1, RSS_FETCH_MAX_RETRIES + 1):
        try:
            r = s.get(RSS_URL, timeout=RSS_FETCH_TIMEOUT)
            status = r.status_code
            b = r.content
            r.close()
            if status >= 400 or not b:
                raise RuntimeError(f"RSS_HTTP_{status}_EMPTY={not bool(b)}")
            return b.decode("utf-8", errors="replace")
        except Exception as e:
            last_err = e
            base = min(15.0, 0.8 * (2 ** (attempt - 1)))
            time.sleep(base * (1.0 + random.random() * 0.25))
    raise RuntimeError(f"RSS fetch failed after retries: {last_err}")

# -----------------------------
# RSS pubDate parsing
# -----------------------------
def parse_pubdate_safe(s: str) -> tuple[datetime | None, str]:
    raw = (s or "").strip()
    if not raw:
        return None, raw
    try:
        dt = parsedate_to_datetime(raw)
        if dt is not None:
            return dt.replace(tzinfo=None), raw
    except Exception:
        pass
    for fmt in ("%m/%d/%Y  %I:%M:%S %p", "%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M"):
        try:
            return datetime.strptime(raw, fmt), raw
        except Exception:
            pass
    return None, raw

def split_desc_subject(desc: str) -> tuple[str, str]:
    desc = desc or ""
    if "|SUBJECT:" in desc:
        d, s = desc.split("|SUBJECT:", 1)
        return d.strip(), s.strip()
    return desc.strip(), ""

# -----------------------------
# RSS parsing (ElementTree → regex fallback)
# -----------------------------
def _tag_endswith(el, name: str) -> bool:
    return (el.tag or "").lower().endswith(name.lower())

def parse_rss_items(xml_text: str) -> list[dict]:
    out = []

    try:
        root = ET.fromstring(xml_text)
        for item in root.iter():
            if not _tag_endswith(item, "item"):
                continue

            def get_child(tag_name: str) -> str:
                for ch in list(item):
                    if _tag_endswith(ch, tag_name):
                        return (ch.text or "").strip()
                return ""

            title = get_child("title")
            link = get_child("link").strip()
            desc = get_child("description")
            pub_raw = get_child("pubDate")
            guid = get_child("guid")

            if not link:
                continue

            pub_dt, pub_raw_norm = parse_pubdate_safe(pub_raw)
            description, subject = split_desc_subject(desc)

            out.append({
                "Title": title,
                "Link": link,
                "PubDate": pub_raw_norm,
                "pub_dt": pub_dt,
                "Description": description,
                "Subject": subject,
                "GUID": guid,
            })

        if out:
            return out
    except Exception:
        pass

    items = re.findall(r"<item\b.*?</item>", xml_text, flags=re.DOTALL | re.IGNORECASE)

    def extract(tag, block):
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", block, flags=re.DOTALL | re.IGNORECASE)
        if not m:
            return ""
        val = m.group(1)
        val = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", val, flags=re.DOTALL)
        return html.unescape(val).strip()

    for block in items:
        title = extract("title", block)
        link = extract("link", block).strip()
        if not link:
            continue
        desc = extract("description", block)
        pub_raw = extract("pubDate", block)
        guid = extract("guid", block)

        pub_dt, pub_raw_norm = parse_pubdate_safe(pub_raw)
        description, subject = split_desc_subject(desc)

        out.append({
            "Title": title,
            "Link": link,
            "PubDate": pub_raw_norm,
            "pub_dt": pub_dt,
            "Description": description,
            "Subject": subject,
            "GUID": guid,
        })

    return out

# -----------------------------
# Dedup by Title + time clustering (prefer XML in cluster)
# -----------------------------
def dt_sort_key(d: datetime | None) -> datetime:
    return d if d is not None else datetime.min

def dedup_by_title_time(rows: list[dict]) -> list[dict]:
    by_title = defaultdict(list)
    for r in rows:
        by_title[r.get("Title", "")].append(r)

    final = []
    for _, group in by_title.items():
        group.sort(key=lambda x: dt_sort_key(x.get("pub_dt")))

        cluster = [group[0]]
        for r in group[1:]:
            d_prev = cluster[-1].get("pub_dt")
            d_cur = r.get("pub_dt")

            if d_prev is None or d_cur is None:
                xmls = [x for x in cluster if is_xml_url(x.get("Link", ""))]
                pick = max(xmls or cluster, key=lambda x: dt_sort_key(x.get("pub_dt")))
                final.append(pick)
                cluster = [r]
                continue

            if abs((d_cur - d_prev).total_seconds()) <= CLUSTER_SECONDS:
                cluster.append(r)
            else:
                xmls = [x for x in cluster if is_xml_url(x.get("Link", ""))]
                pick = max(xmls or cluster, key=lambda x: dt_sort_key(x.get("pub_dt")))
                final.append(pick)
                cluster = [r]

        xmls = [x for x in cluster if is_xml_url(x.get("Link", ""))]
        pick = max(xmls or cluster, key=lambda x: dt_sort_key(x.get("pub_dt")))
        final.append(pick)

    return final

# -----------------------------
# Dedup by Link (keep newest PubDate; tie-break prefer XML)
# -----------------------------
def dedup_by_link_keep_newest(rows: list[dict]) -> list[dict]:
    best = {}
    for r in rows:
        link = r.get("Link", "")
        if not link:
            continue
        cur = best.get(link)
        if cur is None:
            best[link] = r
            continue

        d_new = dt_sort_key(r.get("pub_dt"))
        d_old = dt_sort_key(cur.get("pub_dt"))

        if d_new > d_old:
            best[link] = r
        elif d_new == d_old:
            if is_xml_url(link) and not is_xml_url(cur.get("Link", "")):
                best[link] = r

    return list(best.values())

# -----------------------------
# Dedup by identical extracted Content (keep earliest pub_dt)
# -----------------------------
def dedup_by_content_keep_earliest_pubdt(rows: list[dict], results: dict) -> list[dict]:
    best_by_content = {}  # content -> (dt_or_max, link)
    keep_links = set()

    for r in rows:
        link = r.get("Link", "")
        status, content = results.get(link, ("Not Extracted", ""))
        if not content:
            keep_links.add(link)
            continue

        d = r.get("pub_dt") or datetime.max
        cur = best_by_content.get(content)

        if cur is None:
            best_by_content[content] = (d, link)
        else:
            d_prev, link_prev = cur
            if d < d_prev:
                best_by_content[content] = (d, link)
            elif d == d_prev:
                if is_xml_url(link) and not is_xml_url(link_prev):
                    best_by_content[content] = (d, link)

    for _, link in best_by_content.values():
        keep_links.add(link)

    return [r for r in rows if r.get("Link", "") in keep_links]

# -----------------------------
# Shared text normalization
# -----------------------------
def _normalize_for_match(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\x0c", "\n")
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[\u2000-\u200B]", " ", s)
    return s

_space2 = re.compile(r"[ ]{2,}")
_space1 = re.compile(r"[ \t]+")

def normalize_single_cell(text: str, *, tableish: bool) -> str:
    if not text:
        return ""
    s = _normalize_for_match(text)

    out_lines = []
    for ln in s.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        if tableish:
            ln = _space2.sub("\\t", ln)
            ln = _space1.sub(" ", ln)
        out_lines.append(ln)

    compact = "\n".join(out_lines).strip().replace("\n", "\\n")
    if len(compact) > CONTENT_CAP:
        compact = compact[:CONTENT_CAP] + "...[TRUNCATED]"
    return compact

# -----------------------------
# XML decode + strict trim (XBRL)
# -----------------------------
def maybe_decompress_blob(b: bytes) -> bytes:
    if not b:
        return b
    if len(b) >= 2 and b[0:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(b)
        except Exception:
            return b
    if len(b) >= 4 and b[0:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                names = z.namelist()
                if not names:
                    return b
                xml_names = [n for n in names if n.lower().endswith(".xml")]
                pick = xml_names[0] if xml_names else names[0]
                return z.read(pick)
        except Exception:
            return b
    return b

_enc_re = re.compile(r'encoding=["\']([A-Za-z0-9_\-]+)["\']', re.IGNORECASE)

def decode_xml_bytes(b: bytes) -> str:
    if not b:
        return ""
    if b.startswith(b"\xff\xfe") or b.startswith(b"\xfe\xff"):
        try:
            return b.decode("utf-16", errors="replace")
        except Exception:
            pass
    if b.startswith(b"\xef\xbb\xbf"):
        b = b[3:]

    head = b[:512].decode("ascii", errors="ignore")
    m = _enc_re.search(head)
    if m:
        enc = m.group(1).strip()
        try:
            return b.decode(enc, errors="replace")
        except Exception:
            pass

    try:
        return b.decode("utf-8", errors="replace")
    except Exception:
        return b.decode("latin-1", errors="replace")

XBRL_START_RE = re.compile(
    r"<\s*in-capmkt\s*:\s*(NSESymbol|NameOfTheCompany)\b[^>]*>",
    re.IGNORECASE
)
XBRL_END_RE_STRICT = re.compile(r"</\s*xbrli\s*:\s*xbrl\s*>", re.IGNORECASE)
XBRL_END_RE_FALLBACK = re.compile(r"</\s*xbrl\s*>", re.IGNORECASE)

def trim_xbrl_payload(text: str) -> str:
    if not text:
        return ""
    s = text

    m_start = XBRL_START_RE.search(s)
    if not m_start:
        return ""

    start_pos = m_start.start()

    end_match = None
    for m in XBRL_END_RE_STRICT.finditer(s, start_pos):
        end_match = m
    if end_match is None:
        for m in XBRL_END_RE_FALLBACK.finditer(s, start_pos):
            end_match = m
    if end_match is None:
        return ""

    end_pos = end_match.start()
    if end_pos <= start_pos:
        return ""
    return s[start_pos:end_pos].strip()

def extract_xml_from_bytes(b: bytes) -> str:
    b2 = maybe_decompress_blob(b)
    txt = decode_xml_bytes(b2)

    trimmed = trim_xbrl_payload(txt)
    if trimmed:
        return trimmed

    try:
        unescaped = html.unescape(txt)
    except Exception:
        unescaped = ""
    if unescaped and unescaped != txt:
        trimmed2 = trim_xbrl_payload(unescaped)
        if trimmed2:
            return trimmed2

    return ""

def detect_doc_type(url: str, b: bytes) -> str:
    u = (url or "").lower().strip()
    if is_pdf_url(u):
        return "pdf"
    if is_xml_url(u):
        return "xml"

    b2 = maybe_decompress_blob(b)
    head = (b2 or b"").lstrip()[:64]
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(b"<?xml") or head.lower().startswith(b"<xbrl") or head.startswith(b"<"):
        return "xml"
    return "other"

# -----------------------------
# PDF extraction rules
# -----------------------------
START_RE = re.compile(
    r"(?is)("
    r"\bsubject\b|"
    r"\bsubiect\b|"
    r"\bsubict\b|"
    r"\bsub\b|"
    r"\bre\s*[\.: -]|"
    r"\bdear\b|"
    r"\bregulation\b|"
    r"\breg\s*[\.:]|"
    r"\breg\b"
    r")"
)
END_COMPSEC_RE = re.compile(r"(?is)\bcompany\s+secre(?:tary|tery)\b")
END_DIN_RE     = re.compile(r"(?is)\(din\b|\bdin\b")
END_FALLBACK_RE = re.compile(r"(?is)\b(thanking\s+you|yours\s+faithfully|yours\s+truly)\b")
CLOSE_ANCHOR_RE = re.compile(r"(?is)\b(yours\s+(faithfully|sincerely|truly)|thanking\s+you|thank\s+you)\b")

def extract_span_single(text: str) -> str:
    s = _normalize_for_match(text)
    if not s:
        return ""
    m_start = START_RE.search(s)
    if not m_start:
        return ""
    tail = s[m_start.start():]
    if not tail.strip():
        return ""

    cs_all = list(END_COMPSEC_RE.finditer(tail))
    din_all = list(END_DIN_RE.finditer(tail))

    if cs_all or din_all:
        close_matches = list(CLOSE_ANCHOR_RE.finditer(tail))
        anchor = close_matches[-1].start() if close_matches else 0
        cs_after = END_COMPSEC_RE.search(tail, anchor)
        din_after = END_DIN_RE.search(tail, anchor)

        if cs_after or din_after:
            if cs_after and din_after:
                chosen_kind, chosen_match = ("cs", cs_after) if cs_after.start() <= din_after.start() else ("din", din_after)
            elif cs_after:
                chosen_kind, chosen_match = "cs", cs_after
            else:
                chosen_kind, chosen_match = "din", din_after
        else:
            last_cs = cs_all[-1] if cs_all else None
            last_din = din_all[-1] if din_all else None
            if last_cs and last_din:
                chosen_kind, chosen_match = ("cs", last_cs) if last_cs.start() >= last_din.start() else ("din", last_din)
            elif last_cs:
                chosen_kind, chosen_match = "cs", last_cs
            else:
                chosen_kind, chosen_match = "din", last_din

        if chosen_kind == "din":
            return tail[:chosen_match.start()].strip()

        end_idx = chosen_match.end()
        nl = tail.find("\n", end_idx)
        end_idx = nl if nl != -1 else len(tail)
        return tail[:end_idx].strip()

    m_fb = END_FALLBACK_RE.search(tail)
    if not m_fb:
        return ""
    end_idx = m_fb.end()
    nl = tail.find("\n", end_idx)
    end_idx = nl if nl != -1 else len(tail)
    return tail[:end_idx].strip()

def extract_pdf_span_text(pdf_bytes: bytes, max_pages: int) -> tuple[str, int]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        buf = ""
        total_chars = 0
        started = False

        for pi in range(min(len(doc), max_pages)):
            t = doc[pi].get_text("text") or ""
            t_stripped = t.strip()
            total_chars += len(t_stripped)
            if not t_stripped:
                continue

            buf += "\n" + t
            if not started and START_RE.search(buf):
                started = True

            if started:
                span = extract_span_single(buf)
                if span:
                    return span, total_chars

            if not started and len(buf) > 240_000:
                buf = buf[-80_000:]

        span = extract_span_single(buf)
        return span, total_chars
    finally:
        doc.close()

def extract_pdf_span_ocr(pdf_bytes: bytes, max_pages: int, scale: float, tesseract_config: str) -> str:
    if not OCR_ENABLED:
        return ""

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        buf = ""
        started = False

        for pi in range(min(len(doc), max_pages)):
            page = doc[pi]
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)

            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img = ImageOps.grayscale(img)
            img = ImageOps.autocontrast(img)

            ocr_txt = pytesseract.image_to_string(img, lang="eng", config=tesseract_config) or ""
            if not ocr_txt.strip():
                continue

            buf += "\n" + ocr_txt
            if not started and START_RE.search(buf):
                started = True

            if started:
                span = extract_span_single(buf)
                if span:
                    return span.strip()

            if not started and len(buf) > 240_000:
                buf = buf[-80_000:]

        return extract_span_single(buf)
    finally:
        doc.close()

# -----------------------------
# Per-row processing
# -----------------------------
def process_one_row(row: dict) -> tuple[str, str, bytes]:
    link = (row.get("Link") or "").strip()
    if not link:
        return "Not Extracted", "", b""

    ok, b = fetch_bytes(link)
    if not ok or not b:
        return "Not Extracted", "", b""

    try:
        doc_type = detect_doc_type(link, b)

        if doc_type == "xml":
            trimmed = extract_xml_from_bytes(b)
            if not trimmed:
                return "Not Extracted", "", b""
            return "XML extraction", normalize_single_cell(trimmed, tableish=False), b""

        if doc_type == "pdf":
            span, total_chars = extract_pdf_span_text(b, PDF_TEXT_MAX_PAGES)
            if span:
                return "PDF extraction", normalize_single_cell(span, tableish=True), b

            if OCR_ENABLED and total_chars < OCR_TEXTCHAR_THRESHOLD:
                span2 = extract_pdf_span_ocr(b, OCR_MAX_PAGES, OCR_SCALE, "--oem 1 --psm 6")
                if span2:
                    return "OCR extraction", normalize_single_cell(span2, tableish=True), b

            return "Not Extracted", "", b

        return "Not Extracted", "", b""

    except Exception:
        if detect_doc_type(link, b) == "pdf":
            return "Not Extracted", "", b
        return "Not Extracted", "", b""

def retry_pdf_text(pdf_bytes: bytes) -> str:
    try:
        span, _ = extract_pdf_span_text(pdf_bytes, PDF_TEXT_RETRY_MAX_PAGES)
        return span.strip() if span else ""
    except Exception:
        return ""

def retry_pdf_ocr(pdf_bytes: bytes) -> str:
    if not OCR_ENABLED:
        return ""
    try:
        span = extract_pdf_span_ocr(pdf_bytes, OCR_RETRY_MAX_PAGES, OCR_RETRY_SCALE, OCR_RETRY_TESS_CONFIG)
        return span.strip() if span else ""
    except Exception:
        return ""

def retry_xml_link(url: str) -> str:
    for attempt in range(1, XML_RETRY_ATTEMPTS + 1):
        ok, b = fetch_bytes(url)
        if ok and b:
            trimmed = extract_xml_from_bytes(b)
            if trimmed:
                return trimmed.strip()
        backoff(attempt)
    return ""

def chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

# -----------------------------
# One run: RSS text -> output CSV
# -----------------------------
def run_pipeline_from_rss_text(rss_text: str, out_csv_path: Path):
    m = re.search(r"<rss\b", rss_text, re.IGNORECASE)
    if m:
        rss_text = rss_text[m.start():]

    records = parse_rss_items(rss_text)
    print("Parsed items:", len(records))

    dedup_title = dedup_by_title_time(records)
    print("After title/time dedup:", len(dedup_title))

    unique_links = sorted({r["Link"] for r in dedup_title if r.get("Link")})
    print("Unique links (precheck set):", len(unique_links))

    exists_map = {}
    total = len(unique_links)
    checked = 0
    print(f"Exists precheck: batch_size={EXISTS_BATCH_SIZE}, workers={EXISTS_MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=EXISTS_MAX_WORKERS) as ex:
        for batch in chunked(unique_links, EXISTS_BATCH_SIZE):
            futs = {ex.submit(exists_check, u): u for u in batch}
            for fut in as_completed(futs):
                u = futs[fut]
                try:
                    ok = bool(fut.result())
                except Exception:
                    ok = True
                exists_map[u] = ok

            checked += len(batch)
            if checked % 100 == 0 or checked == total:
                print(f"  checked {checked}/{total}")

            time.sleep(EXISTS_BATCH_PAUSE_S * (1.0 + random.random() * 0.20))

    exists_true_rows = [r for r in dedup_title if exists_map.get(r.get("Link", ""), False)]
    print("Rows with Exists=TRUE (after title/time dedup):", len(exists_true_rows))

    to_process = dedup_by_link_keep_newest(exists_true_rows)
    to_process.sort(key=lambda x: x.get("Link", ""))
    print("Unique Exists=TRUE links to process:", len(to_process), f"(workers={PROCESS_MAX_WORKERS})")

    results = {}
    pdf_cache = {}
    print("Phase 1: extraction (MuPDF + gated OCR; strict XBRL trim) ...")

    with ThreadPoolExecutor(max_workers=PROCESS_MAX_WORKERS) as ex:
        futs = {ex.submit(process_one_row, r): r["Link"] for r in to_process}
        done = 0
        for fut in as_completed(futs):
            link = futs[fut]
            try:
                status, content, pdf_bytes = fut.result()
            except Exception:
                status, content, pdf_bytes = ("Not Extracted", "", b"")

            results[link] = (status, content)
            if status == "Not Extracted" and pdf_bytes:
                pdf_cache[link] = pdf_bytes

            done += 1
            if done % 25 == 0:
                print(f"  processed {done}/{len(to_process)}")

    xml_retry_targets = [link for link, (st, _) in results.items() if st == "Not Extracted" and is_xml_url(link)]
    print(f"Phase 2: Not Extracted XML links to retry: {len(xml_retry_targets)}")

    if xml_retry_targets:
        workers = max(1, min(XML_RETRY_WORKERS, PROCESS_MAX_WORKERS))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(retry_xml_link, link): link for link in xml_retry_targets}
            done = 0
            for fut in as_completed(futs):
                link = futs[fut]
                try:
                    trimmed = fut.result() or ""
                except Exception:
                    trimmed = ""
                if trimmed:
                    results[link] = ("XML extraction", normalize_single_cell(trimmed, tableish=False))
                done += 1
                if done % 25 == 0:
                    print(f"  xml retried {done}/{len(xml_retry_targets)}")

    retry_targets = [link for link, (st, _) in results.items() if st == "Not Extracted" and link in pdf_cache]
    print(f"Phase 3: Not Extracted PDFs to retry: {len(retry_targets)}")

    TEXT_RETRY_WORKERS = min(PROCESS_MAX_WORKERS, 4)
    print(f"Phase 3a: PDF text retry (workers={TEXT_RETRY_WORKERS}) ...")

    with ThreadPoolExecutor(max_workers=TEXT_RETRY_WORKERS) as ex:
        futs = {ex.submit(retry_pdf_text, pdf_cache[link]): link for link in retry_targets}
        done = 0
        for fut in as_completed(futs):
            link = futs[fut]
            try:
                span = fut.result() or ""
            except Exception:
                span = ""
            if span:
                results[link] = ("PDF extraction", normalize_single_cell(span, tableish=True))
            done += 1
            if done % 25 == 0:
                print(f"  text retried {done}/{len(retry_targets)}")

    remaining = [link for link in retry_targets if results.get(link, ("Not Extracted", ""))[0] == "Not Extracted"]
    print(f"Phase 3b: OCR retry remaining: {len(remaining)}")

    OCR_RETRY_WORKERS = max(1, min(2, PROCESS_MAX_WORKERS))
    with ThreadPoolExecutor(max_workers=OCR_RETRY_WORKERS) as ex:
        futs = {ex.submit(retry_pdf_ocr, pdf_cache[link]): link for link in remaining}
        done = 0
        for fut in as_completed(futs):
            link = futs[fut]
            try:
                span = fut.result() or ""
            except Exception:
                span = ""
            if span:
                results[link] = ("OCR extraction", normalize_single_cell(span, tableish=True))
            done += 1
            if done % 25 == 0:
                print(f"  ocr retried {done}/{len(remaining)}")

    to_write = dedup_by_content_keep_earliest_pubdt(to_process, results)
    removed = len(to_process) - len(to_write)
    print(f"Phase 4: Content dedup: {len(to_process)} -> {len(to_write)} (removed {removed})")

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Title", "Link", "PubDate", "Description", "Subject", "Exists", "Status", "Content"])
        for r in to_write:
            link = r.get("Link", "")
            status, content = results.get(link, ("Not Extracted", ""))
            w.writerow([
                r.get("Title", ""),
                link,
                r.get("PubDate", ""),
                r.get("Description", ""),
                r.get("Subject", ""),
                "TRUE",
                status,
                content,
            ])

    vals = []
    for r in to_write:
        link = r.get("Link", "")
        st, _ = results.get(link, ("Not Extracted", ""))
        vals.append(st)

    print("\nDONE")
    print("Final rows written:", len(to_write))
    print("PDF extraction:", sum(1 for v in vals if v == "PDF extraction"))
    print("OCR extraction:", sum(1 for v in vals if v == "OCR extraction"))
    print("XML extraction:", sum(1 for v in vals if v == "XML extraction"))
    print("Not Extracted:", sum(1 for v in vals if v == "Not Extracted"))
    print("CSV path:", str(out_csv_path))

def main():
    print("WORKDIR:", str(WORKDIR))
    print("Now (IST):", now_ist().strftime("%Y-%m-%d %H:%M:%S"))
    print("OCR enabled:", OCR_ENABLED)

    label_day = run_label_date_ist()

    rss_text = fetch_rss_xml()
    rss_path = WORKDIR / f"rss_{label_day}.xml"
    rss_path.write_text(rss_text, encoding="utf-8", errors="replace")
    print("RSS fetched and saved:", str(rss_path))

    out_csv = WORKDIR / f"nse_filings_{label_day}.csv"
    run_pipeline_from_rss_text(rss_text, out_csv)

if __name__ == "__main__":
    main()
