from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import datetime as dt
from pathlib import Path
from typing import Any, Callable

from .config import PipelineConfig
from .io_utils import read_json, write_json
from .models import Paper, QueryVariant


def clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_title(title: str) -> str:
    value = clean_text(title).casefold()
    value = re.sub(r"[^\w\u4e00-\u9fff]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def normalize_doi(value: str | None) -> str:
    doi = (value or "").strip().casefold()
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)


def safe_year(value: Any) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


def abstract_from_index(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions = [(int(pos), token) for token, locs in index.items() for pos in locs]
    return " ".join(token for _, token in sorted(positions))


class CachedHTTP:
    def __init__(self, cache_dir: Path, timeout: int = 45) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        contact = os.getenv("LITREVIEW_CONTACT_EMAIL", "course-demo@example.invalid")
        self.user_agent = f"LitReviewAgent/1.0 (mailto:{contact})"
        self.request_count = 0
        self.cache_hits = 0
        self.network_failures = 0
        self.requests: list[dict[str, Any]] = []

    def _path(self, url: str, suffix: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.{suffix}"

    def get_bytes(
        self, url: str, *, headers: dict[str, str] | None = None, cache: bool = True
    ) -> bytes:
        path = self._path(url, "bin")
        if cache and path.exists():
            self.cache_hits += 1
            self.requests.append(self._request_record(url, cache_hit=True))
            return path.read_bytes()
        self.request_count += 1
        request_headers = {"User-Agent": self.user_agent}
        request_headers.update(headers or {})
        request = urllib.request.Request(url, headers=request_headers)
        last: Exception | None = None
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = response.read()
                if cache:
                    path.write_bytes(data)
                self.requests.append(self._request_record(url, cache_hit=False))
                return data
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                last = exc
                self.network_failures += 1
                if attempt < 3:
                    time.sleep(attempt)
        raise RuntimeError(f"GET failed: {url}: {last}")

    def _request_record(self, url: str, *, cache_hit: bool) -> dict[str, Any]:
        parsed = urllib.parse.urlsplit(url)
        params = urllib.parse.parse_qs(parsed.query)
        return {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "endpoint": f"{parsed.scheme}://{parsed.netloc}{parsed.path}",
            "params": {key: values for key, values in sorted(params.items())},
            "cache_hit": cache_hit,
            "url_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
        }

    def get_json(
        self, url: str, *, headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        return json.loads(self.get_bytes(url, headers=headers).decode("utf-8", "replace"))

    def get_text(self, url: str) -> str:
        return self.get_bytes(url).decode("utf-8", "replace")


def _paper(**kwargs: Any) -> Paper:
    kwargs["title"] = clean_text(kwargs.get("title", ""))
    kwargs["abstract"] = clean_text(kwargs.get("abstract", ""))
    kwargs["doi"] = normalize_doi(kwargs.get("doi", ""))
    kwargs["authors"] = [clean_text(x) for x in kwargs.get("authors", []) if clean_text(x)]
    return Paper(**kwargs)


def search_openalex(
    http: CachedHTTP, query: str, limit: int, config: PipelineConfig, round_index: int
) -> list[Paper]:
    params: dict[str, str] = {"search": query, "per-page": str(min(limit, 100))}
    filters = []
    if config.year_from:
        filters.append(f"from_publication_date:{config.year_from}-01-01")
    if config.year_to:
        filters.append(f"to_publication_date:{config.year_to}-12-31")
    if filters:
        params["filter"] = ",".join(filters)
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    data = http.get_json(url, headers={"Accept": "application/json"})
    rows = []
    for item in data.get("results", []):
        location = item.get("best_oa_location") or item.get("primary_location") or {}
        source = (item.get("primary_location") or {}).get("source") or {}
        rows.append(
            _paper(
                paper_id="",
                title=item.get("title", ""),
                authors=[
                    a.get("author", {}).get("display_name", "")
                    for a in item.get("authorships", [])
                ],
                year=item.get("publication_year"),
                abstract=abstract_from_index(item.get("abstract_inverted_index")),
                venue=source.get("display_name", ""),
                sources=["openalex"],
                source_ids={"openalex": item.get("id", "")},
                doi=item.get("doi", ""),
                url=item.get("id", ""),
                open_access_pdf=location.get("pdf_url", "") or "",
                citation_count=int(item.get("cited_by_count") or 0),
                publication_type=str(item.get("type") or ""),
                language=str(item.get("language") or ""),
                is_retracted=bool(item.get("is_retracted", False)),
                retraction_status=(
                    "retracted" if item.get("is_retracted") else "not_flagged_by_openalex"
                ),
                matched_queries=[query],
                retrieval_round=round_index,
            )
        )
    return rows


def search_semantic_scholar(
    http: CachedHTTP, query: str, limit: int, config: PipelineConfig, round_index: int
) -> list[Paper]:
    fields = (
        "paperId,title,authors,year,venue,abstract,url,citationCount,"
        "influentialCitationCount,externalIds,openAccessPdf"
    )
    params = {"query": query, "limit": str(min(limit, 100)), "fields": fields}
    if config.year_from or config.year_to:
        params["year"] = f"{config.year_from or ''}-{config.year_to or ''}"
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search?"
        + urllib.parse.urlencode(params)
    )
    headers = {"Accept": "application/json"}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    data = http.get_json(url, headers=headers)
    rows = []
    for item in data.get("data", []):
        external = item.get("externalIds") or {}
        oa = item.get("openAccessPdf") or {}
        rows.append(
            _paper(
                paper_id="",
                title=item.get("title", ""),
                authors=[a.get("name", "") for a in item.get("authors", [])],
                year=item.get("year"),
                abstract=item.get("abstract", ""),
                venue=item.get("venue", ""),
                sources=["semantic_scholar"],
                source_ids={"semantic_scholar": item.get("paperId", "")},
                doi=external.get("DOI", ""),
                arxiv_id=external.get("ArXiv", ""),
                url=item.get("url", ""),
                open_access_pdf=oa.get("url", "") or "",
                citation_count=int(item.get("citationCount") or 0),
                influential_citation_count=int(
                    item.get("influentialCitationCount") or 0
                ),
                publication_type="preprint" if external.get("ArXiv") else "",
                matched_queries=[query],
                retrieval_round=round_index,
            )
        )
    return rows


def search_crossref(
    http: CachedHTTP, query: str, limit: int, config: PipelineConfig, round_index: int
) -> list[Paper]:
    filters = []
    if config.year_from:
        filters.append(f"from-pub-date:{config.year_from}")
    if config.year_to:
        filters.append(f"until-pub-date:{config.year_to}")
    params: dict[str, str] = {
        "query.bibliographic": query,
        "rows": str(min(limit, 100)),
    }
    if filters:
        params["filter"] = ",".join(filters)
    mail = os.getenv("LITREVIEW_CONTACT_EMAIL")
    if mail:
        params["mailto"] = mail
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    data = http.get_json(url, headers={"Accept": "application/json"})
    rows = []
    for item in data.get("message", {}).get("items", []):
        date_parts = item.get(
            "published-print",
            item.get("published-online", item.get("created", {})),
        ).get("date-parts", [[]])
        year = date_parts[0][0] if date_parts and date_parts[0] else None
        authors = [
            " ".join([author.get("given", ""), author.get("family", "")]).strip()
            for author in item.get("author", [])
        ]
        rows.append(
            _paper(
                paper_id="",
                title=" ".join(item.get("title") or []),
                authors=authors,
                year=year,
                abstract=item.get("abstract", ""),
                venue=" ".join(item.get("container-title") or []),
                sources=["crossref"],
                source_ids={"crossref": item.get("URL", "")},
                doi=item.get("DOI", ""),
                url=item.get("URL", ""),
                citation_count=int(item.get("is-referenced-by-count") or 0),
                publication_type=str(item.get("type") or ""),
                language=str(item.get("language") or ""),
                retraction_status=(
                    "has_update_or_correction"
                    if item.get("update-to") or item.get("relation")
                    else "not_flagged_by_crossref"
                ),
                matched_queries=[query],
                retrieval_round=round_index,
            )
        )
    return rows


def search_arxiv(
    http: CachedHTTP, query: str, limit: int, config: PipelineConfig, round_index: int
) -> list[Paper]:
    del config
    params = {
        "search_query": f'all:"{query}"',
        "start": "0",
        "max_results": str(min(limit, 100)),
        "sortBy": "relevance",
    }
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    root = ET.fromstring(http.get_text(url))
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    rows = []
    for entry in root.findall("atom:entry", ns):
        entry_url = entry.findtext("atom:id", default="", namespaces=ns)
        arxiv_id = entry_url.rstrip("/").split("/")[-1]
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
        rows.append(
            _paper(
                paper_id="",
                title=entry.findtext("atom:title", default="", namespaces=ns),
                authors=[
                    a.findtext("atom:name", default="", namespaces=ns)
                    for a in entry.findall("atom:author", ns)
                ],
                year=safe_year(
                    entry.findtext("atom:published", default="", namespaces=ns)
                ),
                abstract=entry.findtext(
                    "atom:summary", default="", namespaces=ns
                ),
                venue="arXiv",
                sources=["arxiv"],
                source_ids={"arxiv": arxiv_id},
                arxiv_id=arxiv_id,
                url=entry_url,
                open_access_pdf=pdf_url,
                matched_queries=[query],
                retrieval_round=round_index,
                publication_type="preprint",
                version_group_id=re.sub(r"v\d+$", "", arxiv_id),
                version_status="preprint",
            )
        )
    return rows


SEARCHERS: dict[
    str, Callable[[CachedHTTP, str, int, PipelineConfig, int], list[Paper]]
] = {
    "openalex": search_openalex,
    "semantic_scholar": search_semantic_scholar,
    "crossref": search_crossref,
    "arxiv": search_arxiv,
}


def merge_papers(papers: list[Paper], existing: list[Paper] | None = None) -> list[Paper]:
    merged: dict[str, Paper] = {}
    order: list[str] = []
    for paper in (existing or []) + papers:
        title_key = normalize_title(paper.title)
        if not title_key:
            continue
        key = f"doi:{normalize_doi(paper.doi)}" if paper.doi else f"title:{title_key}"
        if key not in merged:
            merged[key] = paper
            order.append(key)
            continue
        current = merged[key]
        current.sources = sorted(set(current.sources + paper.sources))
        current.source_ids.update(paper.source_ids)
        current.matched_queries = sorted(
            set(current.matched_queries + paper.matched_queries)
        )
        for field in (
            "abstract",
            "venue",
            "doi",
            "arxiv_id",
            "url",
            "open_access_pdf",
        ):
            if not getattr(current, field) and getattr(paper, field):
                setattr(current, field, getattr(paper, field))
        if len(paper.abstract) > len(current.abstract):
            current.abstract = paper.abstract
        if len(paper.authors) > len(current.authors):
            current.authors = paper.authors
        current.citation_count = max(current.citation_count, paper.citation_count)
        current.influential_citation_count = max(
            current.influential_citation_count, paper.influential_citation_count
        )
        current.retrieval_round = min(
            current.retrieval_round, paper.retrieval_round
        )
        if current.year and paper.year and current.year != paper.year:
            conflict = f"year:{current.year}|{paper.year}"
            if conflict not in current.metadata_conflicts:
                current.metadata_conflicts.append(conflict)
        if current.venue and paper.venue and current.venue != paper.venue:
            conflict = f"venue:{current.venue}|{paper.venue}"
            if conflict not in current.metadata_conflicts:
                current.metadata_conflicts.append(conflict)
        if paper.is_retracted is True:
            current.is_retracted = True
            current.retraction_status = paper.retraction_status

    result = [merged[key] for key in order]
    seen_ids: set[str] = set()
    for paper in result:
        if paper.paper_id and paper.paper_id in seen_ids:
            paper.paper_id = ""
        if paper.paper_id:
            seen_ids.add(paper.paper_id)
    used_ids = set(seen_ids)
    next_id = 1
    for paper in result:
        if paper.paper_id:
            continue
        while f"P{next_id:04d}" in used_ids:
            next_id += 1
        paper.paper_id = f"P{next_id:04d}"
        used_ids.add(paper.paper_id)
        next_id += 1
    for paper in result:
        if not paper.version_group_id:
            paper.version_group_id = (
                "doi:" + normalize_doi(paper.doi)
                if paper.doi
                else "title:" + normalize_title(paper.title)
            )
        if paper.version_status == "unknown":
            paper.version_status = "published" if paper.doi else "unverified"
    return result


def retrieval_rank(paper: Paper, topic: str) -> float:
    topic_tokens = set(tokenize(topic))
    hay_tokens = set(tokenize(paper.title + " " + paper.abstract))
    overlap = len(topic_tokens & hay_tokens) / max(1, len(topic_tokens))
    source_bonus = len(paper.sources) * 0.35
    citation_bonus = math.log1p(paper.citation_count) * 0.12
    abstract_bonus = 0.3 if paper.abstract else 0
    return overlap * 5 + source_bonus + citation_bonus + abstract_bonus


def tokenize(text: str) -> list[str]:
    lowered = clean_text(text).casefold()
    latin = re.findall(r"[a-z][a-z0-9-]{1,}", lowered)
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]+", lowered)
    cjk = [
        chunk[i : i + 2]
        for chunk in chinese_chunks
        for i in range(max(1, len(chunk) - 1))
        if len(chunk[i : i + 2]) == 2
    ]
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "using",
        "study",
        "analysis",
        "研究",
        "基于",
        "一种",
    }
    return [token for token in latin + cjk if token not in stop]


class SearchManager:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.http = CachedHTTP(
            config.output_dir / "_cache" / "http", timeout=config.request_timeout
        )

    def citation_chase(
        self, seeds: list[Paper], *, round_index: int = 0
    ) -> tuple[list[Paper], dict[str, str]]:
        """Retrieve forward/backward OpenAlex neighbors for a few seed papers."""
        if self.config.offline_corpus or not self.config.enable_citation_chasing:
            return [], {}
        records: list[Paper] = []
        errors: dict[str, str] = {}
        for seed in seeds[: self.config.citation_seed_count]:
            openalex_id = seed.source_ids.get("openalex", "").rstrip("/").split("/")[-1]
            if not openalex_id:
                continue
            try:
                records.extend(
                    _search_openalex_filter(
                        self.http,
                        f"cites:{openalex_id}",
                        self.config.citation_chase_limit,
                        self.config,
                        round_index,
                        f"citation-forward:{seed.paper_id}",
                    )
                )
                work = self.http.get_json(
                    f"https://api.openalex.org/works/{openalex_id}?select=referenced_works"
                )
                referenced = [
                    str(value).rstrip("/").split("/")[-1]
                    for value in work.get("referenced_works", [])
                ][: self.config.citation_chase_limit]
                if referenced:
                    records.extend(
                        _search_openalex_filter(
                            self.http,
                            "openalex:" + "|".join(referenced),
                            self.config.citation_chase_limit,
                            self.config,
                            round_index,
                            f"citation-backward:{seed.paper_id}",
                        )
                    )
            except Exception as exc:
                errors[f"citation_chase:{seed.paper_id}"] = str(exc)
        return merge_papers(records), errors

    def search(
        self, queries: list[QueryVariant], *, round_index: int
    ) -> tuple[list[Paper], dict[str, str]]:
        if self.config.offline_corpus:
            all_rows = [
                Paper.from_dict(item) for item in read_json(self.config.offline_corpus)
            ]
            has_round_labels = any(row.retrieval_round != 0 for row in all_rows)
            rows = (
                [row for row in all_rows if row.retrieval_round == round_index]
                if has_round_labels
                else all_rows
            )
            for row in rows:
                if not row.matched_queries:
                    row.matched_queries = [q.query for q in queries]
            return merge_papers(rows), {}

        records: list[Paper] = []
        errors: dict[str, str] = {}
        for variant in queries:
            for source in self.config.sources:
                searcher = SEARCHERS.get(source)
                if not searcher:
                    errors[f"{source}:{variant.query}"] = "unknown source"
                    continue
                try:
                    records.extend(
                        searcher(
                            self.http,
                            variant.query,
                            self.config.per_query_source_limit,
                            self.config,
                            round_index,
                        )
                    )
                except Exception as exc:  # source failures should not abort the run
                    errors[f"{source}:{variant.query}"] = str(exc)
                time.sleep(0.12)
        papers = merge_papers(records)
        papers.sort(key=lambda p: retrieval_rank(p, self.config.topic), reverse=True)
        return papers, errors

    def save_raw_snapshot(
        self, path: Path, papers: list[Paper], errors: dict[str, str]
    ) -> None:
        write_json(
            path,
            {"papers": [paper.to_dict() for paper in papers], "errors": errors},
        )


def _search_openalex_filter(
    http: CachedHTTP,
    filter_value: str,
    limit: int,
    config: PipelineConfig,
    round_index: int,
    provenance: str,
) -> list[Paper]:
    params: dict[str, str] = {"filter": filter_value, "per-page": str(min(limit, 100))}
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    data = http.get_json(url, headers={"Accept": "application/json"})
    rows: list[Paper] = []
    for item in data.get("results", []):
        location = item.get("best_oa_location") or item.get("primary_location") or {}
        source = (item.get("primary_location") or {}).get("source") or {}
        rows.append(
            _paper(
                paper_id="",
                title=item.get("title", ""),
                authors=[a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])],
                year=item.get("publication_year"),
                abstract=abstract_from_index(item.get("abstract_inverted_index")),
                venue=source.get("display_name", ""),
                sources=["openalex"],
                source_ids={"openalex": item.get("id", "")},
                doi=item.get("doi", ""),
                url=item.get("id", ""),
                open_access_pdf=location.get("pdf_url", "") or "",
                citation_count=int(item.get("cited_by_count") or 0),
                publication_type=str(item.get("type") or ""),
                language=str(item.get("language") or ""),
                is_retracted=bool(item.get("is_retracted", False)),
                retraction_status="retracted" if item.get("is_retracted") else "not_flagged_by_openalex",
                matched_queries=[provenance],
                retrieval_round=round_index,
            )
        )
    return rows
