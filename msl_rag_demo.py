#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "medical_knowledge_base.jsonl"

DISCLAIMER = "仅用于RAG流程演示；真实医学答复需由医学/合规人员基于获批资料审核。"

SYNONYMS = {
    "副作用": ["不良反应", "安全性", "风险"],
    "不良反应": ["副作用", "安全性", "风险"],
    "联用": ["合用", "相互作用", "一起用"],
    "一起用": ["联用", "合用", "相互作用"],
    "能不能": ["可以", "是否", "禁忌", "慎用"],
    "怀孕": ["妊娠", "孕晚期", "孕妇"],
    "孕晚期": ["妊娠", "怀孕"],
    "肾功能": ["eGFR", "肾损伤", "肾功能不全"],
    "造影": ["含碘造影", "造影剂"],
    "肌痛": ["肌无力", "CK", "横纹肌溶解"],
    "过量": ["中毒", "用药过量"],
    "证据": ["文献", "研究", "荟萃分析"],
    "剂量": ["用法用量", "每日", "维持剂量"],
    "MSL": ["医学事务", "医学答复", "合规"],
}

PRODUCT_ALIASES = {
    "阿司匹林": ["阿司匹林", "aspirin"],
    "奥美拉唑": ["奥美拉唑", "omeprazole"],
    "二甲双胍": ["二甲双胍", "metformin"],
    "瑞舒伐他汀": ["瑞舒伐他汀", "rosuvastatin"],
    "氯吡格雷": ["氯吡格雷", "clopidogrel"],
}

INTENT_KEYWORDS = {
    "不良反应": ["副作用", "不良反应", "安全", "风险", "肌痛", "出血", "低镁"],
    "药物相互作用": ["联用", "合用", "一起用", "相互作用", "华法林", "布洛芬", "奥美拉唑", "吉非贝齐"],
    "禁忌与慎用": ["禁忌", "慎用", "能不能", "可以吗", "孕晚期", "哮喘", "溃疡"],
    "特殊人群": ["肾功能", "eGFR", "妊娠", "怀孕", "哺乳", "老年", "儿童"],
    "用法用量": ["剂量", "用法", "每日", "二级预防", "服用"],
    "过量处理": ["过量", "中毒", "耳鸣", "酸中毒"],
    "文献证据": ["文献", "研究", "证据", "疗效", "获益", "PCI", "DAPT"],
    "合规答复": ["MSL", "合规", "医学答复", "升级", "SOP"],
}


@dataclass(frozen=True)
class Document:
    id: str
    product: str
    source_type: str
    section: str
    last_updated: str
    tags: tuple[str, ...]
    content: str


@dataclass(frozen=True)
class SearchHit:
    doc: Document
    score: float
    matched_terms: tuple[str, ...]


def load_documents(path: Path = DATA_PATH) -> list[Document]:
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            documents.append(
                Document(
                    id=item["id"],
                    product=item["product"],
                    source_type=item["source_type"],
                    section=item["section"],
                    last_updated=item["last_updated"],
                    tags=tuple(item["tags"]),
                    content=item["content"],
                )
            )
    return documents


def cjk_ngrams(text: str) -> Iterable[str]:
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    for char in chars:
        yield char
    for size in (2, 3, 4):
        for index in range(len(chars) - size + 1):
            yield "".join(chars[index : index + size])


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    normalized = text.lower()
    tokens.extend(re.findall(r"[a-z0-9]+(?:[./+-][a-z0-9]+)*", normalized))
    tokens.extend(cjk_ngrams(normalized))
    return [token for token in tokens if token.strip()]


def expand_query(query: str) -> str:
    expanded = [query]
    for word, replacements in SYNONYMS.items():
        if word.lower() in query.lower():
            expanded.extend(replacements)
    return " ".join(expanded)


class LocalRAG:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.doc_tokens: list[Counter[str]] = []
        self.idf: dict[str, float] = {}
        self._build_index()

    def _build_index(self) -> None:
        document_frequency: Counter[str] = Counter()
        for doc in self.documents:
            searchable_text = " ".join(
                [doc.product, doc.source_type, doc.section, " ".join(doc.tags), doc.content]
            )
            counts = Counter(tokenize(searchable_text))
            self.doc_tokens.append(counts)
            document_frequency.update(counts.keys())

        total = len(self.documents)
        self.idf = {
            token: math.log((total + 1) / (df + 0.5)) + 1.0
            for token, df in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        expanded_query = expand_query(query)
        query_counts = Counter(tokenize(expanded_query))
        if not query_counts:
            return []

        hits: list[SearchHit] = []
        query_text = expanded_query.lower()
        requested_products = detect_requested_products(query)
        egfr_value = extract_egfr_value(query)
        for doc, doc_counts in zip(self.documents, self.doc_tokens):
            score = 0.0
            matched_terms: set[str] = set()
            doc_length = sum(doc_counts.values()) or 1

            for token, query_tf in query_counts.items():
                if token not in doc_counts:
                    continue
                tf = doc_counts[token] / doc_length
                score += (1.0 + math.log1p(query_tf)) * tf * self.idf.get(token, 1.0) * 100.0
                if len(token) > 1:
                    matched_terms.add(token)

            metadata = " ".join([doc.product, doc.section, " ".join(doc.tags)]).lower()
            for phrase in re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]+", query_text):
                if phrase and phrase in metadata:
                    score += 3.5
                    matched_terms.add(phrase)
                elif phrase and phrase in doc.content.lower():
                    score += 1.5
                    matched_terms.add(phrase)

            doc_metadata = " ".join([doc.product, doc.section, " ".join(doc.tags), doc.content]).lower()
            if requested_products:
                if document_matches_products(doc_metadata, requested_products):
                    score += 14.0
                else:
                    score -= 18.0

            if egfr_value is not None:
                if 30 <= egfr_value <= 44 and re.search(r"30\s*至\s*44|30\s*-\s*44", doc.content):
                    score += 22.0
                    matched_terms.add("eGFR 30-44")
                if "造影" in doc_metadata and "造影" not in query:
                    score -= 10.0

            if score > 0:
                hits.append(SearchHit(doc=doc, score=score, matched_terms=tuple(sorted(matched_terms))))

        sorted_hits = sorted(hits, key=lambda hit: hit.score, reverse=True)
        if not sorted_hits:
            return []
        minimum_score = max(8.0, sorted_hits[0].score * 0.35)
        return [hit for hit in sorted_hits if hit.score >= minimum_score][:top_k]

    def answer(self, query: str, top_k: int = 5) -> str:
        hits = self.search(query, top_k=top_k)
        if not hits:
            return "\n".join(
                [
                    f"医生问题：{query}",
                    "",
                    "未在当前知识库中检索到足够相关的内容。",
                    #f"提示：{DISCLAIMER}",
                ]
            )

        intent = detect_intent(query)
        answer_points = synthesize_points(query, hits)

        lines = [
            f"医生问题：{query}",
            f"识别问题维度：{intent}",
            "",
            "相关内容：",
        ]
        lines.extend([f"- {point}" for point in answer_points])
        lines.extend(
            [
                "",
                "检索依据：",
            ]
        )
        for index, hit in enumerate(hits, start=1):
            excerpt = best_excerpt(query, hit.doc.content)
            lines.append(
                f"[{index}] {hit.doc.product}｜{hit.doc.source_type}｜{hit.doc.section}｜"
                f"{hit.doc.last_updated}｜score={hit.score:.2f}"
            )
            lines.append(f"    {excerpt}")
        #lines.extend(["", f"合规提示：{DISCLAIMER}"])
        return "\n".join(lines)


def detect_intent(query: str) -> str:
    scores: dict[str, int] = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        scores[intent] = sum(1 for keyword in keywords if keyword.lower() in query.lower())
    best_intent, best_score = max(scores.items(), key=lambda item: item[1])
    return best_intent if best_score > 0 else "综合医学信息咨询"


def detect_requested_products(query: str) -> set[str]:
    normalized = query.lower()
    requested: set[str] = set()
    for product, aliases in PRODUCT_ALIASES.items():
        if any(alias.lower() in normalized for alias in aliases):
            requested.add(product)
    return requested


def document_matches_products(doc_metadata: str, requested_products: set[str]) -> bool:
    for product in requested_products:
        aliases = PRODUCT_ALIASES[product]
        if any(alias.lower() in doc_metadata for alias in aliases):
            return True
    return False


def extract_egfr_value(query: str) -> int | None:
    match = re.search(r"egfr\s*([0-9]{1,3})", query, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def split_sentences(text: str) -> list[str]:
    raw_sentences = re.split(r"(?<=[。！？；])", text)
    return [sentence.strip() for sentence in raw_sentences if sentence.strip()]


def best_excerpt(query: str, content: str, max_chars: int = 160) -> str:
    query_terms = set(token for token in tokenize(expand_query(query)) if len(token) > 1)
    sentences = split_sentences(content)
    if not sentences:
        return content[:max_chars]

    def score(sentence: str) -> int:
        sentence_tokens = set(tokenize(sentence))
        return len(query_terms & sentence_tokens)

    selected = max(sentences, key=score)
    if len(selected) <= max_chars:
        return selected
    return selected[: max_chars - 1] + "..."


def synthesize_points(query: str, hits: list[SearchHit], max_points: int = 4) -> list[str]:
    query_terms = set(token for token in tokenize(expand_query(query)) if len(token) > 1)
    candidates: list[tuple[int, str]] = []
    seen: set[str] = set()
    intent = detect_intent(query)
    egfr_value = extract_egfr_value(query)
    top_score = hits[0].score if hits else 1.0

    for hit in hits:
        for sentence in split_sentences(hit.doc.content):
            sentence_tokens = set(tokenize(sentence))
            overlap = len(query_terms & sentence_tokens)
            section_bonus = 2 if any(term in hit.doc.section for term in ["相互作用", "不良反应", "禁忌", "剂量", "证据", "SOP"]) else 0
            intent_bonus = 0
            if intent == "不良反应" and any(term in hit.doc.section for term in ["不良反应", "安全性"]):
                intent_bonus += 5
            if intent == "药物相互作用" and "相互作用" in hit.doc.section:
                intent_bonus += 5
            if intent in {"特殊人群", "禁忌与慎用", "用法用量"} and any(
                term in hit.doc.section for term in ["肾功能", "特殊人群", "禁忌", "剂量", "用法"]
            ):
                intent_bonus += 5
            if intent == "文献证据" and any(term in hit.doc.section for term in ["证据", "文献", "实践"]):
                intent_bonus += 5
            if egfr_value is not None and 30 <= egfr_value <= 44 and re.search(r"30\s*至\s*44|30\s*-\s*44", sentence):
                intent_bonus += 8
            if "造影" in sentence and "造影" not in query:
                intent_bonus -= 4

            hit_bonus = round((hit.score / top_score) * 3)
            score = overlap + section_bonus + intent_bonus + hit_bonus
            normalized = re.sub(r"\s+", "", sentence)
            if score > 0 and normalized not in seen:
                candidates.append((score, sentence.rstrip("。；")))
                seen.add(normalized)

    candidates.sort(key=lambda item: item[0], reverse=True)
    points = [sentence for _, sentence in candidates[:max_points]]

    if not points:
        points = [best_excerpt(query, hit.doc.content, max_chars=120).rstrip("。；") for hit in hits[:max_points]]

    #points.append("建议MSL在发送前核对患者适应证、合并用药、肝肾功能、出血/缺血风险，并保留来源引用")
    return points


def run_demo(rag: LocalRAG) -> None:
    demo_questions = [
        "阿司匹林有哪些常见副作用？哪些情况需要警惕？",
        "氯吡格雷和奥美拉唑联用会不会影响抗血小板效果？",
        "eGFR 35 的患者还能不能用二甲双胍？",
        "瑞舒伐他汀患者出现肌痛和深色尿，需要怎么评估？",
        "PCI 术后阿司匹林联合氯吡格雷需要关注什么？",
    ]
    for number, question in enumerate(demo_questions, start=1):
        print("=" * 88)
        print(f"Demo Case {number}")
        print(rag.answer(question, top_k=4))
        print()


def interactive_loop(rag: LocalRAG) -> None:
    #print("MSL 医学问答 RAG Demo")
    print("输入问题后回车；输入 exit 退出。")
    print(f"知识库：{DATA_PATH}")
    while True:
        try:
            query = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue
        print()
        print(rag.answer(query))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local MSL medical affairs RAG demo")
    parser.add_argument("--question", "-q", help="医生提出的医学问题")
    parser.add_argument("--demo", action="store_true", help="运行内置的多场景演示问题")
    parser.add_argument("--top-k", type=int, default=5, help="返回的检索片段数量")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_documents()
    rag = LocalRAG(documents)

    if args.demo:
        run_demo(rag)
    elif args.question:
        print(rag.answer(args.question, top_k=args.top_k))
    else:
        interactive_loop(rag)


if __name__ == "__main__":
    main()
