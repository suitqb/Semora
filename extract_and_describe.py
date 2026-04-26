    """
    extract_and_describe.py — Benchmark VLM pour description d'images

    1. Charge les résultats Docling (zones picture) depuis un JSON Docling natif
    2. Croppe les zones image depuis les PDFs sources (coordonnées Docling : origine bas-gauche)
    3. Envoie chaque image aux VLMs configurés
    4. Sauvegarde les résultats dans vlm_descriptions.json

    Usage :
        python extract_and_describe.py --pdfs pdfs/ --docling_jsons extract_docling/ --output results/vlm_benchmark/
        python extract_and_describe.py --images_dir results/vlm_benchmark/images/ --output results/vlm_benchmark/

    Variables d'environnement (.env) :
        OPENAI_API_KEY         — clé pour gpt-4o-mini et gpt-5-mini
        OPENAI_API_BASE        — base URL pour gpt-4o-mini et gpt-5-mini
        MISTRAL_LARGE_API_BASE — base URL pour mistral-large
        MISTRAL_API_KEY        — clé pour mistral-medium
        MISTRAL_API_BASE       — base URL pour mistral-medium
    """

    import argparse
    import base64
    import json
    import os
    import time
    from io import BytesIO
    from pathlib import Path

    import fitz
    from dotenv import load_dotenv
    from loguru import logger
    from PIL import Image

    load_dotenv()

    # ══════════════════════════════════════════════════════════════════════════
    #  Config modèles
    # ══════════════════════════════════════════════════════════════════════════

    # Modèles OpenAI-compat : (model_id, env var de la base URL)
    _OPENAI_MODELS = {
        "mistral-large": ("ADADAS-mistral-large-3", "MISTRAL_LARGE_API_BASE"),
        "gpt-4o-mini":   ("gpt-4o-mini",            "OPENAI_API_BASE"),
        "gpt-5-mini":    ("ADADAS-gpt-5-mini",       "OPENAI_API_BASE"),
    }

    MODELS = ["mistral-medium", "mistral-large", "gpt-4o-mini", "gpt-5-mini"]

    # ══════════════════════════════════════════════════════════════════════════
    #  Prompts
    # ══════════════════════════════════════════════════════════════════════════

    SYSTEM_PROMPT = """Tu es un assistant technique spécialisé en documentation industrielle.
    Tu analyses des images extraites de gammes d'assemblage et de maintenance de systèmes automatisés."""

    USER_PROMPT = """Décris précisément cette image issue d'une gamme d'assemblage ou de maintenance industrielle.
    Ta description doit être exploitable dans un système RAG (recherche documentaire).

    Inclure obligatoirement :
    - Les éléments visibles (pièces, composants, équipements)
    - Les actions ou gestes représentés si applicable
    - Les annotations, flèches, légendes présentes dans l'image
    - La référence ou le nom des pièces si lisible
    - L'orientation ou le positionnement spatial des éléments

    Sois précis et factuel. Ne suppose pas ce qui n'est pas visible."""

    # ══════════════════════════════════════════════════════════════════════════
    #  Appel VLM
    # ══════════════════════════════════════════════════════════════════════════

    def describe(image_b64: str, model_name: str) -> tuple[str, float]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]},
        ]

        t0 = time.perf_counter()

        if model_name == "mistral-medium":
            from mistralai.client import MistralClient
            client = MistralClient(
                api_key=os.environ["MISTRAL_API_KEY"],
                server_url=os.environ["MISTRAL_API_BASE"],
            )
            resp = client.chat.complete(
                model="mistral-medium-latest",
                messages=messages,
                max_tokens=600,
                temperature=0.2,
            )
            return resp.choices[0].message.content, time.perf_counter() - t0

        model_id, base_url_env = _OPENAI_MODELS[model_name]
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ[base_url_env],
            timeout=60,
        )
        # max_completion_tokens requis par les modèles récents (gpt-5-mini+),
        # fallback sur max_tokens pour les endpoints plus anciens
        try:
            resp = client.chat.completions.create(
                model=model_id, messages=messages,
                max_completion_tokens=600, temperature=0.2,
            )
        except Exception as e:
            if "max_completion_tokens" in str(e) or "unsupported_parameter" in str(e):
                resp = client.chat.completions.create(
                    model=model_id, messages=messages,
                    max_tokens=600, temperature=0.2,
                )
            else:
                raise
        return resp.choices[0].message.content, time.perf_counter() - t0


    # ══════════════════════════════════════════════════════════════════════════
    #  Extraction Docling / PDF
    # ══════════════════════════════════════════════════════════════════════════

    def load_docling_picture_zones(docling_json_path: Path) -> list[dict]:
        with open(docling_json_path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and data.get("schema_name") == "DoclingDocument":
            zones = []
            for pic in data.get("pictures", []):
                provs = pic.get("prov", [])
                if not provs:
                    continue
                prov = provs[0]
                bbox, page_num = prov.get("bbox", {}), prov.get("page_no")
                if not page_num or not bbox:
                    continue
                zones.append({
                    "page":    page_num,
                    "x0":      bbox.get("l", 0),
                    "y0":      bbox.get("b", 0),
                    "x1":      bbox.get("r", 0),
                    "y1":      bbox.get("t", 0),
                    "zone_id": pic.get("self_ref", ""),
                })
            logger.info(f"  {len(zones)} zones picture — {docling_json_path.name}")
            return zones

        if isinstance(data, list):
            zones = [z for z in data if str(z.get("zone_type", "")).lower() == "picture"]
            logger.info(f"  {len(zones)} zones picture — {docling_json_path.name}")
            return zones

        logger.warning(f"  Format JSON non reconnu : {docling_json_path.name}")
        return []


    def crop_image_from_pdf(
        pdf_path: Path, page_num: int,
        bbox: tuple[float, float, float, float],
        padding: float = 5.0, dpi: int = 150,
    ) -> "Image.Image | None":
        l, b, r, t = bbox
        try:
            with fitz.open(str(pdf_path)) as doc:
                page = doc[page_num - 1]
                ph = page.rect.height
                clip = fitz.Rect(max(0, l - padding), max(0, (ph - t) - padding),
                                r + padding, (ph - b) + padding)
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=clip)
                return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        except Exception as e:
            logger.error(f"  Erreur crop page {page_num} : {e}")
            return None


    def extract_images_from_docling(
        pdf_paths: list[Path], docling_json_dir: Path, output_dir: Path,
    ) -> list[dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        images_meta = []

        for pdf_path in pdf_paths:
            candidates = (
                list(docling_json_dir.glob(f"sortie_{pdf_path.stem}.json")) or
                list(docling_json_dir.glob(f"{pdf_path.stem}_docling*.json")) or
                list(docling_json_dir.glob(f"*{pdf_path.stem}*.json"))
            )
            if not candidates:
                logger.warning(f"Pas de JSON Docling pour {pdf_path.name}, ignoré")
                continue

            logger.info(f"Traitement : {pdf_path.name} → {candidates[0].name}")
            for i, zone in enumerate(load_docling_picture_zones(candidates[0]), start=1):
                img = crop_image_from_pdf(pdf_path, zone["page"],
                                        (zone["x0"], zone["y0"], zone["x1"], zone["y1"]))
                if img is None or img.width < 50 or img.height < 50:
                    continue
                fname = f"{pdf_path.stem}_p{zone['page']}_img{i:02d}.png"
                img.save(output_dir / fname)
                images_meta.append({
                    "image_id":   fname.replace(".png", ""),
                    "image_path": str(output_dir / fname),
                    "doc":        pdf_path.name,
                    "page":       zone["page"],
                    "zone_id":    zone.get("zone_id", ""),
                    "width":      img.width,
                    "height":     img.height,
                })
                logger.info(f"  ✓ {fname} ({img.width}×{img.height})")

        logger.info(f"Total : {len(images_meta)} images extraites")
        return images_meta


    # ══════════════════════════════════════════════════════════════════════════
    #  Encodage base64
    # ══════════════════════════════════════════════════════════════════════════

    def image_to_base64(img_path: Path, max_size: int = 1024) -> str:
        img = Image.open(img_path).convert("RGB")
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


    # ══════════════════════════════════════════════════════════════════════════
    #  Orchestration
    # ══════════════════════════════════════════════════════════════════════════

    def run_benchmark(images_meta: list[dict], output_dir: Path) -> list[dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for meta in images_meta:
            logger.info(f"\n── {meta['image_id']} ({meta['doc']}, p{meta['page']}) ──")
            image_b64 = image_to_base64(Path(meta["image_path"]))
            row = {**meta, "descriptions": {}}

            for model_name in MODELS:
                logger.info(f"  → {model_name}...")
                try:
                    text, duration = describe(image_b64, model_name)
                    row["descriptions"][model_name] = {
                        "text": text, "duration_s": round(duration, 2), "error": None,
                    }
                    logger.info(f"     ✓ {duration:.1f}s — {len(text)} chars")
                except Exception as e:
                    logger.error(f"     ✗ {e}")
                    row["descriptions"][model_name] = {
                        "text": "", "duration_s": None, "error": str(e),
                    }
                time.sleep(0.5)

            results.append(row)

        json_path = output_dir / "vlm_descriptions.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"\nRésultats sauvegardés : {json_path}")
        return results


    # ══════════════════════════════════════════════════════════════════════════
    #  Point d'entrée
    # ══════════════════════════════════════════════════════════════════════════

    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--pdfs",          type=Path)
        parser.add_argument("--docling_jsons", type=Path)
        parser.add_argument("--images_dir",    type=Path)
        parser.add_argument("--output",        type=Path, default=Path("results/vlm_benchmark"))
        args = parser.parse_args()

        images_dir = args.output / "images"

        if args.images_dir:
            images_meta = [
                {
                    "image_id":   p.stem,
                    "image_path": str(p),
                    "doc":        p.stem.rsplit("_p", 1)[0] + ".pdf" if "_p" in p.stem else p.name,
                    "page":       int(p.stem.split("_p")[1].split("_")[0]) if "_p" in p.stem else 0,
                    "zone_id": "", "width": 0, "height": 0,
                }
                for p in sorted(args.images_dir.glob("*.png"))
            ]
        elif args.pdfs and args.docling_jsons:
            pdf_paths = sorted(args.pdfs.glob("*.pdf"))
            logger.info(f"{len(pdf_paths)} PDFs trouvés")
            images_meta = extract_images_from_docling(pdf_paths, args.docling_jsons, images_dir)
        else:
            parser.error("Fournir soit --pdfs + --docling_jsons, soit --images_dir")
            return

        if not images_meta:
            logger.error("Aucune image à traiter.")
            return

        logger.info(f"{len(images_meta)} images · {len(MODELS)} modèles")
        run_benchmark(images_meta, args.output)


    if __name__ == "__main__":
        main()
