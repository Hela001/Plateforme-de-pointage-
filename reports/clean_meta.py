import json

# Chemin vers ton ancien meta.json
input_file = "meta.json"
output_file = "meta_clean.json"

# Charger l'ancien fichier
with open(input_file, "r", encoding="utf-8") as f:
    old_data = json.load(f)

# Vérifier si c'est un dictionnaire avec des clés numériques
if isinstance(old_data, dict):
    # Convertir en liste
    reports = []
    seen_ids = set()  # Pour éviter les doublons
    for key in sorted(old_data.keys(), key=int):
        report = old_data[key]
        if report["report_id"] not in seen_ids:
            reports.append(report)
            seen_ids.add(report["report_id"])
else:
    reports = old_data  # Si c'était déjà une liste

# Créer le nouveau dictionnaire
new_data = {"reports": reports}

# Sauvegarder dans un nouveau fichier
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print(f"Fichier nettoyé et sauvegardé dans '{output_file}' avec {len(reports)} rapports.")
