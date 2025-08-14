import re
import csv
import pdfplumber
import pandas as pd
from typing import List, Dict, Tuple
import os
import json
import requests


PAN_REGEX = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
NAME_HINT = re.compile(r"\b(Name|Applicant|Assessee|Authorized Signatory|Director|Proprietor)\b", re.I)
ORG_HINT = re.compile(r"\b(Company|Limited|Ltd\.?|LLP|Pvt\.?|Private|Inc\.?|LLC|Enterprises|Technologies|Solutions|Bank|Institute|University|Trust|Society|Association)\b", re.I)


def extract_text_from_pdf(pdf_path: str) -> str:
    texts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            texts.append(t)
    return "\n".join(texts)


def heuristic_candidates(text: str) -> Dict[str, List[str]]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Collect PANs with high precision regex
    pans = list(dict.fromkeys(PAN_REGEX.findall(text)))

    names: List[str] = []
    orgs: List[str] = []
    for ln in lines:
        if ORG_HINT.search(ln):
            cleaned = re.sub(r"^[\-•\d\.\)\(\s]+", "", ln)
            if 3 <= len(cleaned) <= 120:
                orgs.append(cleaned)
        if NAME_HINT.search(ln):
            cleaned = re.sub(r"^[\-•\d\.\)\(\s]+", "", ln)
            cleaned = re.sub(r"^(Name|Applicant|Assessee|Authorized Signatory|Director|Proprietor)\s*[:\-]\s*", "", cleaned, flags=re.I)
            if 2 <= len(cleaned) <= 120:
                names.append(cleaned)
        tokens = ln.split()
        if 1 < len(tokens) <= 4 and all(t[:1].isupper() for t in tokens if t.isalpha()):
            cand = re.sub(r"[^A-Za-z\s\.&]", "", ln).strip()
            if 3 <= len(cand) <= 80:
                names.append(cand)

    names = list(dict.fromkeys(names))
    orgs = list(dict.fromkeys(orgs))
    return {"PAN": pans, "Name": names, "Organisation": orgs}


def pair_pan_relations(text: str, pans: List[str], names: List[str], orgs: List[str]) -> List[Tuple[str, str, str]]:

    triples: List[Tuple[str, str, str]] = []


    def idx(hay: str, needle: str) -> int:
        try:
            return hay.index(needle)
        except ValueError:
            return -1

    for pan in pans:
        p_pos = idx(text, pan)
        best_name = None
        best_org = None
        best_name_d = 10**9
        best_org_d = 10**9
        for nm in names:
            n_pos = idx(text, nm)
            if n_pos >= 0:
                d = abs(n_pos - p_pos) if p_pos >= 0 else 10**9
                if d < best_name_d:
                    best_name_d = d
                    best_name = nm
        for og in orgs:
            o_pos = idx(text, og)
            if o_pos >= 0:
                d = abs(o_pos - p_pos) if p_pos >= 0 else 10**9
                if d < best_org_d:
                    best_org_d = d
                    best_org = og
        if best_name:
            triples.append((best_name, "PAN_Of", pan))
        if best_org:
            triples.append((best_org, "PAN_Of", pan))
    return triples


def _chunk_text(text: str, max_chars: int = 8000) -> List[str]:

    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: List[str] = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p) if cur else p
        else:
            if cur:
                chunks.append(cur)
            cur = p[:max_chars]
    if cur:
        chunks.append(cur)
    return chunks


def _parse_llm_json(s: str) -> Dict:

    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def _ollama_generate(prompt: str, model: str = "mistral", base_url: str = "http://localhost:11434", temperature: float = 0.0, timeout: float = 120.0) -> str | None:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("response", "")
        return None
    except Exception:
        return None


def llm_candidates(text: str, model: str = "mistral", base_url: str = "http://localhost:11434") -> Dict[str, List[str]]:
    prompt_header = (
        "You are an expert information extractor. Extract entities and relations from the provided text.\n"
        "Output ONLY valid JSON with the following schema: {\n"
        "  \"entities\": { \"Organisation\": [str], \"Name\": [str], \"PAN\": [str] },\n"
        "  \"relations\": [ { \"subject\": str, \"predicate\": \"PAN_Of\", \"object\": str } ]\n"
        "}. Rules: \n"
        "- PAN must match regex [A-Z]{5}[0-9]{4}[A-Z}.\n"
        "- Deduplicate values.\n"
        "- Keep subjects exactly as they appear in text.\n"
        "- Include only predicate \"PAN_Of\".\n"
        "- If nothing found, return empty arrays.\n"
    )

    all_names: List[str] = []
    all_orgs: List[str] = []
    all_pans: List[str] = []

    for chunk in _chunk_text(text, 6000):
        prompt = prompt_header + "\nTEXT:\n" + chunk
        resp = _ollama_generate(prompt, model=model, base_url=base_url)
        if not resp:
            continue
        data = _parse_llm_json(resp)
        ents = data.get("entities", {}) if isinstance(data, dict) else {}
        names = ents.get("Name", []) if isinstance(ents, dict) else []
        orgs = ents.get("Organisation", []) if isinstance(ents, dict) else []
        pans = ents.get("PAN", []) if isinstance(ents, dict) else []
        # Validate PANs
        pans = [p for p in pans if PAN_REGEX.fullmatch(p or "")] 
        all_names.extend([s for s in names if isinstance(s, str)])
        all_orgs.extend([s for s in orgs if isinstance(s, str)])
        all_pans.extend([s for s in pans if isinstance(s, str)])


    def dedup(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen and x:
                seen.add(x)
                out.append(x)
        return out

    return {
        "PAN": dedup(all_pans),
        "Name": dedup(all_names),
        "Organisation": dedup(all_orgs),
    }


def llm_relations(text: str, names: List[str], orgs: List[str], pans: List[str], model: str = "mistral", base_url: str = "http://localhost:11434") -> List[Tuple[str, str, str]]:
    subject_list = names + orgs
    if not subject_list or not pans:
        return []

    prompt = (
        "Link subjects to their PAN using only exact subjects from the provided list and PANs from the list.\n"
        "Return JSON: {\"relations\":[{\"subject\":str,\"predicate\":\"PAN_Of\",\"object\":str}]}.\n"
        "Rules: Use only subjects from SUBJECTS and PANs from PANS; if unknown, skip. No duplicates.\n"
        f"SUBJECTS: {json.dumps(subject_list)}\n"
        f"PANS: {json.dumps(pans)}\n"
        f"TEXT:\n{text}\n"
    )
    resp = _ollama_generate(prompt, model=model, base_url=base_url)
    if not resp:
        return []
    data = _parse_llm_json(resp)
    out: List[Tuple[str, str, str]] = []
    for r in (data.get("relations") or []):
        if not isinstance(r, dict):
            continue
        s = r.get("subject")
        p = r.get("predicate")
        o = r.get("object")
        if p == "PAN_Of" and isinstance(s, str) and isinstance(o, str) and o in pans and s in subject_list:
            out.append((s, p, o))

    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def to_csv(entities: Dict[str, List[str]], relations: List[Tuple[str, str, str]], out_path: str) -> None:
    ent_rows = []
    for etype, vals in entities.items():
        for v in vals:
            ent_rows.append({"type": etype, "value": v})
    ent_df = pd.DataFrame(ent_rows)

    rel_df = pd.DataFrame(relations, columns=["subject", "predicate", "object"])

    base = out_path.rsplit(".", 1)[0]
    ent_df.to_csv(f"{base}_entities.csv", index=False)
    rel_df.to_csv(f"{base}_relations.csv", index=False)

    combined_cols = ["category", "type", "subject", "predicate", "object", "value"]
    ent_comb = ent_df.assign(category="entity", subject=None, predicate=None, object=None)[["category", "type", "subject", "predicate", "object", "value"]]
    rel_comb = rel_df.assign(category="relation", type="PAN_Of", value=None)[["category", "type", "subject", "predicate", "object", "value"]]
    comb_df = pd.concat([ent_comb, rel_comb], ignore_index=True)
    comb_df.to_csv(f"{base}_combined.csv", index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract entities and PAN relations from a PDF")
    parser.add_argument("pdf", help="Path to input PDF")
    parser.add_argument("--out", default="output.csv", help="Output CSV base name (will create _entities and _relations)")
    parser.add_argument("--use-llm", action="store_true", help="Use local Ollama with an open-source model (e.g., mistral)")
    parser.add_argument("--llm-model", default="mistral", help="Ollama model name (e.g., mistral, llama3, etc.)")
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"), help="Base URL for Ollama server")
    args = parser.parse_args()

    text = extract_text_from_pdf(args.pdf)

    heur = heuristic_candidates(text)

    if args.use_llm:
        llm_ents = llm_candidates(text, model=args.llm_model, base_url=args.ollama_url)

        merged = {
            "PAN": list(dict.fromkeys([*heur["PAN"], *llm_ents["PAN"]])),
            "Name": list(dict.fromkeys([*heur["Name"], *llm_ents["Name"]])),
            "Organisation": list(dict.fromkeys([*heur["Organisation"], *llm_ents["Organisation"]])),
        }

        rels_llm = llm_relations(text, merged["Name"], merged["Organisation"], merged["PAN"], model=args.llm_model, base_url=args.ollama_url)
        rels_prox = pair_pan_relations(text, merged["PAN"], merged["Name"], merged["Organisation"]) if not rels_llm else []
        rels = rels_llm or rels_prox
        to_csv(merged, rels, args.out)
    else:

        rels = pair_pan_relations(text, heur["PAN"], heur["Name"], heur["Organisation"]) 
        to_csv(heur, rels, args.out)

    print("Saved:")
    base = args.out.rsplit(".", 1)[0]
    print(f" - {base}_entities.csv")
    print(f" - {base}_relations.csv")
    print(f" - {base}_combined.csv")
