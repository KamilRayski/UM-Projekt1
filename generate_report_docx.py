import json
import shutil
from pathlib import Path

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from config import (
    FIGURES_DIR,
    METRICS_SUMMARY_PATH,
    REPORT_PATH,
    resolve_report_template,
)
from report_metrics import build_dataset_split_rows_docx


def set_cell_shading(cell, color_hex: str):
    shading_elm = OxmlElement("w:shd")
    shading_elm.set(qn("w:val"), "clear")
    shading_elm.set(qn("w:color"), "auto")
    shading_elm.set(qn("w:fill"), color_hex)
    cell._tc.get_or_add_tcPr().append(shading_elm)


def set_table_borders(table, color_hex: str = "BFBFBF"):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for border_name in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        border = OxmlElement(f"w:{border_name}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:space"), "0")
        border.set(qn("w:color"), color_hex)
        borders.append(border)
    tblPr.append(borders)


def add_formatted_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    set_table_borders(table)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    hdr_cells = table.rows[0].cells
    for i, h_text in enumerate(headers):
        hdr_cells[i].text = ""
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h_text)
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_shading(hdr_cells[i], "1F4E79")

    # Data rows
    for r_idx, row_vals in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        bg = "F2F5F9" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            row_cells[c_idx].text = ""
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            r.font.name = "Times New Roman"
            r.font.size = Pt(10)
            if r_idx == len(rows) - 1 and ("PneumoniaCNN" in str(row_vals[0]) or "Razem" in str(row_vals[0])):
                r.bold = True
            set_cell_shading(row_cells[c_idx], bg)

    doc.add_paragraph("")
    return table


def add_body_paragraph(doc, text: str, bold_prefix: str = None):
    p = doc.add_paragraph(style="Normal")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_bold = p.add_run(bold_prefix)
        r_bold.bold = True
        r_bold.font.name = "Times New Roman"
        r_bold.font.size = Pt(11)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    return p


def add_bullet_point(doc, bold_prefix: str, text: str):
    p = doc.add_paragraph(style="Normal")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    r_b = p.add_run("• " + bold_prefix)
    r_b.bold = True
    r_b.font.name = "Times New Roman"
    r_b.font.size = Pt(11)
    r_t = p.add_run(text)
    r_t.font.name = "Times New Roman"
    r_t.font.size = Pt(11)
    return p


def add_figure_with_caption(doc, img_path: Path, caption: str, width_inches: float = 6.2):
    if not img_path.exists():
        return
    p_img = doc.add_paragraph(style="Normal")
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    run_img = p_img.add_run()
    run_img.add_picture(str(img_path), width=Inches(width_inches))

    p_cap = doc.add_paragraph(style="Normal")
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(10)
    r_cap = p_cap.add_run(caption)
    r_cap.italic = True
    r_cap.font.name = "Times New Roman"
    r_cap.font.size = Pt(9.5)
    r_cap.font.color.rgb = RGBColor(60, 60, 60)


def add_heading_1(doc, text: str):
    h = doc.add_paragraph(text, style="Heading 1")
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(6)
    return h


def add_heading_3(doc, text: str):
    h = doc.add_paragraph(text, style="Heading 3")
    h.paragraph_format.space_before = Pt(10)
    h.paragraph_format.space_after = Pt(4)
    return h


def _pl_pct(value: float) -> str:
    return f"{value * 100:.2f}%".replace(".", ",")


def _pl_auc(value: float) -> str:
    return f"{value:.4f}".replace(".", ",")


def _format_param_dict(params: dict) -> str:
    pairs = []
    for key, val in params.items():
        if val is None:
            val_s = "brak limitu" if key == "max_depth" else "None"
        else:
            val_s = str(val)
        pairs.append(f"{key} = {val_s}")
    return ", ".join(pairs)


def build_hyperparam_rows(summary: dict) -> list[list[str]]:
    rows = []
    name_map = {
        "Logistic Regression": "Logistic Regression (PCA)",
        "Random Forest": "Random Forest",
        "SVM (RBF)": "SVM (RBF + PCA)",
        "PneumoniaCNN (PyTorch)": "PneumoniaCNN (PyTorch)",
    }
    search_space = {
        "Logistic Regression": "C ∈ {0.01, 0.1, 1.0, 10.0}, class_weight ∈ {None, 'balanced'}",
        "Random Forest": "n_estimators ∈ {150, 300}, max_depth ∈ {12, 24, None}, class_weight ∈ {'balanced', 'balanced_subsample'}",
        "SVM (RBF)": "C ∈ {0.5, 1.0, 5.0, 10.0}, gamma ∈ {'scale', 0.001, 0.0005}, class_weight ∈ {None, 'balanced'}",
        "PneumoniaCNN (PyTorch)": "Architektura 4-blokowa, Dropout, AdamW lr=1e-3, CosineAnnealingLR, Weighted CE",
    }
    for model_key, info in summary["best_params_summary"].items():
        label = name_map.get(model_key, model_key)
        val_metrics = summary["results_summary"][model_key]["val"]
        rows.append(
            [
                label,
                search_space.get(model_key, "—"),
                _format_param_dict(info.get("best_params", {})),
                f"{info.get('cv_roc_auc', 0):.4f}".replace(".", ",") + f" (Val ROC-AUC: {_pl_auc(val_metrics['roc_auc'])})",
            ]
        )
    return rows


def build_evaluation_rows(summary: dict) -> list[list[str]]:
    rows = []
    specs = [
        ("Logistic Regression", "Walidacyjny, n=524", "val"),
        ("Random Forest", "Walidacyjny, n=524", "val"),
        ("SVM (RBF)", "Walidacyjny, n=524", "val"),
        ("PneumoniaCNN (PyTorch)", "Walidacyjny, n=524", "val"),
        ("Logistic Regression", "Testowy – próg 0,50", "test_default"),
        ("Random Forest", "Testowy – próg 0,50", "test_default"),
        ("SVM (RBF)", "Testowy – próg 0,50", "test_default"),
        ("PneumoniaCNN (PyTorch)", "Testowy – próg 0,50", "test_default"),
        ("Logistic Regression", "Testowy – próg z walidacji (Youden)", "test_val_threshold"),
        ("Random Forest", "Testowy – próg z walidacji (Youden)", "test_val_threshold"),
        ("SVM (RBF)", "Testowy – próg z walidacji (Youden)", "test_val_threshold"),
        ("PneumoniaCNN (PyTorch)", "Testowy – próg z walidacji (Youden)", "test_val_threshold"),
    ]
    short = {
        "Logistic Regression": "Logistic Regression",
        "Random Forest": "Random Forest",
        "SVM (RBF)": "SVM RBF",
        "PneumoniaCNN (PyTorch)": "PneumoniaCNN",
    }
    for model_key, split_label, block in specs:
        m = summary["results_summary"][model_key][block]
        rows.append(
            [
                f"{short[model_key]} ({split_label})",
                _pl_pct(m["accuracy"]),
                _pl_pct(m["balanced_accuracy"]),
                _pl_pct(m["precision"]),
                _pl_pct(m["recall"]),
                _pl_pct(m["specificity"]),
                _pl_pct(m["f1"]),
                _pl_auc(m["roc_auc"]),
            ]
        )
    return rows


def build_report():
    if not METRICS_SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            f"Brak {METRICS_SUMMARY_PATH}. Uruchom najpierw: uv run python run_pipeline.py"
        )
    template_path = resolve_report_template()
    with open(METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    doc = docx.Document(str(template_path))

    # 1. Update Title Page in place (Paragraphs 0..19)
    # Paragraph 11 is 'xxx' under 'Temat Projektu:'
    doc.paragraphs[11].text = ""
    r_title = doc.paragraphs[11].add_run(
        "PneumoniaMNIST – klasyfikacja zdjęć rentgenowskich płuc w kierunku zapalenia płuc"
    )
    r_title.bold = True
    r_title.font.name = "Times New Roman"
    r_title.font.size = Pt(16)

    # Paragraphs 16..19 are 'xxx' under 'Wykonali:'
    authors = ["Kamil Rayski", "Miłosz Pabis", "", ""]
    for idx, author_name in zip([16, 17, 18, 19], authors):
        doc.paragraphs[idx].text = ""
        if author_name:
            r_auth = doc.paragraphs[idx].add_run(author_name)
            r_auth.bold = True
            r_auth.font.name = "Times New Roman"
            r_auth.font.size = Pt(12)

    # Remove template placeholder paragraphs from index 20 onwards so we can populate rich structured content
    for p in list(doc.paragraphs[20:]):
        p._element.getparent().remove(p._element)

    # Page break after title page
    doc.add_page_break()

    # =========================================================================
    # 1. WSTĘP
    # =========================================================================
    add_heading_1(doc, "1. Wstęp")

    add_body_paragraph(
        doc,
        "Głównym celem niniejszego projektu jest opracowanie, wytrenowanie oraz kompleksowa ewaluacja metod "
        "uczenia maszynowego i głębokiego uczenia (Deep Learning) przeznaczonych do automatycznej klasyfikacji "
        "pediatrycznych zdjęć rentgenowskich (RTG) klatki piersiowej w projekcji przednio-tylnej (AP) w kierunku "
        "rozpoznawania zapalenia płuc (Pneumonia). Zapalenie płuc należy do najczęstszych i najgroźniejszych "
        "ostrych infekcji dolnych dróg oddechowych, stanowiąc według danych Światowej Organizacji Zdrowia (WHO) "
        "jedną z głównych przyczyn hospitalizacji oraz śmiertelności wśród dzieci poniżej 5. roku życia. "
        "W praktyce klinicznej podstawowym badaniem obrazowym potwierdzającym obecność zmian zapalnych miąższu "
        "płucnego jest radiogram klatki piersiowej. Interpretacja zdjęć RTG wymaga jednak dużego doświadczenia "
        "radiologicznego, a różnicowanie pomiędzy prawidłowym obrazem płuc (NORMAL) a zacienieniami pęcherzykowymi "
        "(bakteryjnymi) lub śródmiąższowymi (wirusowymi) bywa obarczone subiektywnością oraz ryzykiem błędów "
        "wynikających z przeciążenia personelu medycznego. Zbudowany w ramach projektu system wspomagania "
        "decyzji klinicznych (CDSS – Clinical Decision Support System) ma na celu szybkie i wiarygodne "
        "przesiewowe wykrywanie przypadków patologicznych wraz z wizualnym wskazaniem obszaru zmienionego "
        "chorobowo (wyjaśnialna sztuczna inteligencja – XAI).",
        bold_prefix="Cel projektu: "
    )

    add_heading_3(doc, "Opis danych:")
    add_body_paragraph(
        doc,
        "W projekcie wykorzystano referencyjny zbiór danych klinicznych Chest X-Ray Images (Pneumonia) "
        "opublikowany przez zespół Kermany et al. (University of California San Diego / Guangzhou Women and "
        "Children’s Medical Center), stanowiący również bazę źródłową ustandaryzowanego benchmarku medycznego "
        "PneumoniaMNIST (wchodzącego w skład kolekcji MedMNIST). Zbiór został pobrany programowo przy użyciu "
        "biblioteki kagglehub (identyfikator: paultimothymooney/chest-xray-pneumonia)."
    )

    add_bullet_point(
        doc,
        "Liczba przykładów i podział zbioru: ",
        "Zbiór obejmuje łącznie 5 856 pediatrycznych zdjęć RTG klatki piersiowej wykonanych u pacjentów "
        "w wieku od 1 do 5 lat. W oryginalnym podziale repozytorium Kaggle dane podzielone są na zbiór "
        "treningowy (5 216 zdjęć), walidacyjny (zaledwie 16 zdjęć: 8 NORMAL i 8 PNEUMONIA) oraz testowy "
        "(624 zdjęcia: 234 NORMAL i 390 PNEUMONIA)."
    )
    add_bullet_point(
        doc,
        "Charakterystyka cech i format obrazów: ",
        "Dane wejściowe mają postać dwuwymiarowych obrazów cyfrowych w formacie JPEG (w skali szarości 8-bit 'L' "
        "lub 24-bit 'RGB' z identycznymi kanałami luminancji). Obrazy charakteryzują się zmienną rozdzielczością "
        "natywną: średnia szerokość wynosi 1327,9 ± 363,5 px, natomiast średnia wysokość 970,7 ± 383,4 px."
    )
    add_bullet_point(
        doc,
        "Balans klas i podtypy kliniczne: ",
        "Zbiór wykazuje istotne niezbalansowanie klas – zdjęcia z rozpoznanym zapaleniem płuc (PNEUMONIA) "
        "stanowią 73,0% całego zbioru (4 273 obrazy), podczas gdy przypadki prawidłowe (NORMAL) stanowią 27,0% "
        "(1 583 obrazy). Dodatkowo z analizy metadanych plików klasy PNEUMONIA wynika podział etiologiczny na "
        "bakteryjne zapalenie płuc (2 780 zdjęć – 47,5% ogółu) oraz wirusowe zapalenie płuc (1 493 zdjęcia – 25,5% ogółu)."
    )
    add_bullet_point(
        doc,
        "Zidentyfikowane problemy z jakością danych: ",
        "Podczas inspekcji zidentyfikowano trzy kluczowe wyzwania: (1) krytycznie mały rozmiar oryginalnego "
        "zbioru walidacyjnego (n = 16), uniemożliwiający wiarygodny dobór hiperparametrów i wczesne zatrzymywanie "
        "uczenia; (2) silną dysproporcję liczebności klas w zbiorze treningowym (~3:1 na korzyść klasy PNEUMONIA) "
        "oraz przesunięcie rozkładu a priori w zbiorze testowym (gdzie udział klasy NORMAL rośnie z 25,7% do 37,5%); "
        "(3) obecność artefaktów niemedycznych na zdjęciach RTG (znaczniki radiologiczne 'R'/'L', elektrody EKG, "
        "rurki intubacyjne, zmienna ekspozycja i rotacja tułowia dziecka)."
    )

    add_formatted_table(
        doc,
        headers=["Podzbiór danych", "Oryginalny NORMAL", "Oryginalny PNEUMONIA", "Skorygowany NORMAL", "Skorygowany PNEUMONIA", "Łącznie (po korekcie)"],
        rows=build_dataset_split_rows_docx(summary),
    )

    add_heading_3(doc, "Zakres projektu:")
    add_body_paragraph(
        doc,
        "Zakres zrealizowanych prac badawczo-implementacyjnych obejmuje pełny potok uczenia maszynowego w diagnostyce obrazowej:"
    )
    add_bullet_point(doc, "1. Eksploracyjną analizę danych (EDA): ", "analizę rozkładu klas i podtypów etiologicznych, korektę podziału walidacyjnego zgodnie ze standardem PneumoniaMNIST (4 708 train / 524 val / 624 test), badanie rozkładu rozdzielczości, jasności, kontrastu oraz wyznaczenie uśrednionych szablonów anatomicznych płuc.")
    add_bullet_point(doc, "2. Zaawansowane przetwarzanie i inżynierię cech: ", "ujednolicenie przestrzeni barw i rozdzielczości (64×64 px dla klasycznych algorytmów ML oraz 128×128 px dla sieci konwolucyjnej), ekstrakcję 17 cech domenowych opisujących symetrię pól płucnych i gradienty krawędziowe, standaryzację oraz redukcję wymiarowości metodą PCA (100 składowych głównych – Eigenlungs).")
    add_bullet_point(doc, "3. Budowę i strojenie 4 modeli predykcyjnych: ", "porównanie trzech klasycznych modeli uczenia maszynowego (Regresja Logistyczna, Las Losowy – Random Forest, Maszyna Wektorów Nośnych – SVM z jądrem RBF) z autorską 4-blokową głęboko-konwolucyjną siecią neuronową PneumoniaCNN zaimplementowaną w bibliotece PyTorch.")
    add_bullet_point(doc, "4. Wyjaśnialność (XAI) i analizę błędów klinicznych: ", "wizualizację istotności cech przestrzennych 2D dla modeli klasycznych, generowanie map aktywacji klasowej Grad-CAM dla sieci CNN oraz szczegółową analizę przypadków fałszywie dodatnich (FP) i fałszywie ujemnych (FN) z uwzględnieniem etiologii bakteryjnej i wirusowej.")
    add_bullet_point(doc, "5. Implementację interfejsu użytkownika (GUI): ", "stworzenie interaktywnej aplikacji webowej w technologii Streamlit (app.py), umożliwiającej lekarzowi wgranie dowolnego zdjęcia RTG, porównanie predykcji wszystkich 4 modeli w czasie rzeczywistym oraz wizualizację mapy cieplnej Grad-CAM.")

    # =========================================================================
    # 2. PRZYGOTOWANIE DANYCH
    # =========================================================================
    add_heading_1(doc, "2. Przygotowanie danych")

    add_heading_3(doc, "Eksploracyjna analiza danych (EDA):")
    add_body_paragraph(
        doc,
        "W pierwszym etapie przeprowadzono szczegółową eksploracyjną analizę danych (EDA) na wszystkich 5 856 "
        "obrazach RTG. Jak przedstawiono na Rysunku 1, oryginalny podział zbioru z repozytorium Kaggle posiadał "
        "jedynie 16 zdjęć walidacyjnych. Aby zapewnić rzetelną ocenę modeli w trakcie strojenia hiperparametrów "
        "(i uniknąć wycieku informacji ze zbioru testowego), połączono oryginalne podzbiory train i val, a następnie "
        "dokonano stratyfikowanego, grupowego podziału (StratifiedGroupKFold po identyfikatorze studium/pacjenta "
        "wyciągniętym z nazwy pliku) w proporcji ok. 90% : 10%, uzyskując 4 708 zdjęć treningowych oraz 524 zdjęcia "
        "walidacyjne – co dokładnie odpowiada architekturze "
        "zbioru referencyjnego PneumoniaMNIST. Oryginalny zbiór testowy (n = 624) pozostawiono w pełni nienaruszony."
    )

    add_figure_with_caption(
        doc,
        FIGURES_DIR / "fig1_class_distribution.png",
        "Rysunek 1. Porównanie oryginalnego podziału zbioru Kaggle (lewy wykres), skorygowanego stratyfikowanego podziału zgodnego ze standardem PneumoniaMNIST (środkowy wykres) oraz liczebności podtypów etiologicznych w całym zbiorze (prawy wykres)."
    )

    add_body_paragraph(
        doc,
        "Wizualna inspekcja reprezentatywnych radiogramów z poszczególnych grup klinicznych (Rysunek 2) potwierdza "
        "istotne różnice morfologiczne. Płuca prawidłowe (NORMAL) charakteryzują się wysoką przejrzystością (ciemne "
        "pola płucne wypełnione powietrzem) oraz ostro zarysowanymi kątami przeponowo-żebrowymi i sylwetką serca. "
        "W przypadku bakteryjnego zapalenia płuc widoczne są gęste, ogniskowe lub płatowe zacienienia (konsolidacje "
        "miąższu), natomiast w wirusowym zapaleniu płuc dominuje obustronny, rozlany rysunek śródmiąższowy "
        "o mniejszym kontraście lokalnym, co czyni tę podgrupę znacznie trudniejszą diagnostycznie."
    )

    add_figure_with_caption(
        doc,
        FIGURES_DIR / "fig2_sample_xray_grid.png",
        "Rysunek 2. Przykładowe zdjęcia rentgenowskie klatki piersiowej ze zbioru danych: pacjenci zdrowi (górny rząd), pacjenci z bakteryjnym zapaleniem płuc (środkowy rząd) oraz wirusowym zapaleniem płuc (dolny rząd)."
    )

    add_body_paragraph(
        doc,
        "Analiza rozkładów statystycznych pikseli (Rysunek 3) wykazała, że choć globalna średnia znormalizowana "
        "jasność obrazów w klasach NORMAL (0,481) oraz PNEUMONIA (0,482) jest zbliżona, to rozkład odchylenia "
        "standardowego jasności (kontrastu obrazu) wykazuje wyraźną separację: zdjęcia płuc zdrowych mają wyższy "
        "średni kontrast (0,234) niż zdjęcia z zapaleniem płuc (0,213). Wynika to z faktu, że wysięk zapalny "
        "w pęcherzykach płucnych pochłania promieniowanie rentgenowskie, zmniejszając różnicę gęstości optycznej "
        "pomiędzy upowietrznionym miąższem płuc a śródpiersiem i żebrami. Doskonale obrazuje to mapa różnicowa "
        "uśrednionych obrazów (PNEUMONIA minus NORMAL) widoczna w prawym dolnym rogu Rysunku 3: dodatnie wartości "
        "(kolor czerwony) koncentrują się dokładnie w anatomicznych polach obu płuc, potwierdzając wyższą gęstość "
        "radiologiczną (bielszy obraz) obszarów objętych stanem zapalnym."
    )

    add_figure_with_caption(
        doc,
        FIGURES_DIR / "fig3_image_stats_eda.png",
        "Rysunek 3. Statystyczna eksploracja obrazów: rozkład wymiarów (góra-lewo), gęstość średniej jasności (góra-środek), rozkład kontrastu (góra-prawo) oraz uśrednione szablony płuc dla klas NORMAL, PNEUMONIA i ich mapa różnicowa (dolny rząd)."
    )

    add_heading_3(doc, "Przetwarzanie danych:")
    add_body_paragraph(
        doc,
        "Proces wstępnego przetwarzania danych (preprocessing) został dostosowany do specyfiki dwóch grup modeli:"
    )
    add_bullet_point(
        doc,
        "Konwersja tonalna i ujednolicenie rozdzielczości: ",
        "Wszystkie obrazy skonwertowano do 1-kanałowej skali szarości (tryb 'L') przy użyciu interpolacji "
        "dwuliniowej (Bilinear Resampling). Dla klasycznych modeli ML obrazy przeskalowano do wymiaru 64×64 px "
        "(4 096 cech pikselowych) i znormalizowano do przedziału [0, 1], natomiast dla głębokiej sieci konwolucyjnej "
        "PneumoniaCNN zastosowano wyższą rozdzielczość 128×128 px ze standaryzacją kanału (średnia μ = 0,482, "
        "odchylenie standardowe σ = 0,236 wyznaczone na zbiorze treningowym)."
    )
    add_bullet_point(
        doc,
        "Inżynieria cech domenowych (Feature Engineering): ",
        "Oprócz surowych intensywności pikseli wyekstrahowano zestaw 17 cech radiometrycznych i anatomicznych: "
        "globalną średnią i odchylenie standardowe jasności, rozkład kwantylowy (Q10, Q25, mediana Q50, Q75, Q90), "
        "średnią jasność i wariancję lewego oraz prawego pola płucnego, wskaźnik asymetrii lewo-prawo "
        "|μ_L - μ_P|, średnią gęstość optyczną w strefie górnej, środkowej i dolnej płuc oraz średnią amplitudę "
        "gradientu poziomego i pionowego (opisującą wyrazistość krawędzi żeber i nacieków)."
    )
    add_bullet_point(
        doc,
        "Redukcja wymiarowości metodą PCA (Eigenlungs): ",
        "Ze względu na wysoką współliniowość sąsiednich pikseli obrazu 64×64 (4 096 zmiennych), na ustandaryzowanych "
        "pikselach zbioru treningowego (StandardScaler) dopasowano analizę głównych składowych (PCA). Wybór "
        "k = 100 składowych głównych pozwolił zredukować wymiar przestrzeni pikseli ponad 40-krotnie (z 4 096 do 100), "
        "zachowując jednocześnie aż 91,47% całkowitej wariancji zbioru (Rysunek 4). Wyznaczone wektory własne "
        "(tzw. Eigenlungs) reprezentują globalne wzorce oświetlenia, szerokości klatki piersiowej oraz zacienień "
        "przeponowych i przypodstawnych. Połączony wektor wejściowy dla modeli Regresji Logistycznej oraz SVM "
        "liczył 117 cech (100 składowych PCA + 17 ustandaryzowanych cech domenowych)."
    )
    add_bullet_point(
        doc,
        "Augmentacja danych obrazowych (Data Augmentation): ",
        "W procesie trenowania sieci konwolucyjnej PneumoniaCNN zastosowano losowe transformacje geometryczne "
        "w locie (RandomHorizontalFlip z p = 0,5 oraz RandomAffine: rotacja w zakresie ±8°, przesunięcie do 5% "
        "oraz skalowanie 95%–105%), co zwiększyło odporność sieci na różnice w ułożeniu pacjenta pediatrycznego "
        "podczas wykonywania badania RTG."
    )

    add_figure_with_caption(
        doc,
        FIGURES_DIR / "fig4_pca_analysis.png",
        "Rysunek 4. Analiza głównych składowych (PCA): krzywa skumulowanej wariancji wyjaśnionej (lewo), rzutowanie zbioru treningowego na płaszczyznę PC1 vs PC2 (środek) oraz wizualizacja czterech pierwszych składowych własnych – Eigenlungs (prawo)."
    )

    # =========================================================================
    # 3. MODELOWANIE I EWALUACJA
    # =========================================================================
    add_heading_1(doc, "3. Modelowanie i ewaluacja")

    add_heading_3(doc, "Wybór modeli:")
    add_body_paragraph(
        doc,
        "W celu przeprowadzenia przekrojowej analizy porównawczej wybrano cztery zróżnicowane architektury "
        "reprezentujące zarówno klasyczne paradygmaty uczenia maszynowego (liniowe, drzewiaste, jądrowe), "
        "jak i głębokie sieci konwolucyjne:"
    )
    add_bullet_point(
        doc,
        "1. Regresja Logistyczna (Logistic Regression + PCA): ",
        "Szybki, w pełni interpretowalny liniowy model dyskryminacyjny z regularyzacją L2, stanowiący silny "
        "punkt odniesienia (baseline). Pozwala na bezpośrednie rzutowanie nauczonych wag z powrotem na przestrzeń "
        "2D klatki piersiowej."
    )
    add_bullet_point(
        doc,
        "2. Las Losowy (Random Forest Classifier): ",
        "Nieparametryczny zespół drzew decyzyjnych (bagging) trenowany bezpośrednio na połączonym wektorze "
        "surowych pikseli 64×64 oraz 17 cech domenowych (4 113 zmiennych). Pozwala uchwycić nieliniowe interakcje "
        "pomiędzy regionami płuc oraz wyznaczyć dwuwymiarową mapę istotności pikseli (Gini Importance)."
    )
    add_bullet_point(
        doc,
        "3. Maszyna Wektorów Nośnych (SVM z jądrem RBF + PCA): ",
        "Klasyfikator maksymalizujący margines separacji w nieliniowej przestrzeni cech przekształconej radialną "
        "funkcją bazową (RBF), cechujący się wysoką skutecznością w przestrzeniach o średnim wymiarze (117 cech)."
    )
    add_bullet_point(
        doc,
        "4. Głęboka Sieć Konwolucyjna PneumoniaCNN (PyTorch): ",
        "Autorska 4-blokowa architektura konwolucyjna przetwarzająca obrazy w rozdzielczości 128×128 px. Każdy "
        "z czterech bloków ekstrakcji cech składa się z dwóch warstw splotowych Conv2d (3×3, kolejno 32, 64, 128 "
        "oraz 256 filtrów), normalizacji pakietowej BatchNorm2d, nieliniowości ReLU oraz warstwy redukującej "
        "MaxPool2d(2×2). Zamiast tradycyjnych dużych warstw gęstych na końcu sieci zastosowano warstwę "
        "AdaptiveAvgPool2d(1×1) (Global Average Pooling), co radykalnie redukuje liczbę parametrów, zapobiega "
        "przeuczeniu i umożliwia generowanie map aktywacji Grad-CAM o wymiarze 16×16. Głowica klasyfikująca "
        "zawiera dwie warstwy liniowe (256 → 64 → 2) z regularyzacją Dropout (p = 0,40 oraz p = 0,25)."
    )

    add_heading_3(doc, "Proces trenowania:")
    add_body_paragraph(
        doc,
        "Wszystkie modele trenowano na skorygowanym zbiorze treningowym (n = 4 708). Aby zniwelować wpływ "
        "niezbalansowania klas (74,2% PNEUMONIA vs 25,8% NORMAL), w modelach klasycznych zastosowano ważenie "
        "odwrotnie proporcjonalne do częstości klas (class_weight='balanced' / 'balanced_subsample'), natomiast "
        "w sieci PneumoniaCNN wykorzystano ważoną funkcję straty CrossEntropyLoss z wagami w_0 = 1,939 (dla klasy "
        "NORMAL) oraz w_1 = 0,674 (dla klasy PNEUMONIA). Sieć CNN trenowano przez 15 epok na akceleratorze "
        "Apple Silicon GPU (MPS) z użyciem optymalizatora AdamW (learning rate = 1e-3, weight_decay = 1e-4, "
        "batch_size = 64) oraz harmonogramu kosinusowego CosineAnnealingLR (Rysunek 7). Dodatkowo, z uwagi na "
        "różnicę rozkładu klas i ekspozycji zdjęć pomiędzy zbiorem treningowym a testowym, progi decyzyjne "
        "(Youden J) dobierano wyłącznie na walidacji, a na zbiór testowy nanoszono bez dalszego strojenia. "
        "Prawdopodobieństwa raportowano po skalowaniu Platta (dopasowanym na walidacji)."
    )

    add_figure_with_caption(
        doc,
        FIGURES_DIR / "fig7_cnn_training_curves.png",
        "Rysunek 7. Przebieg procesu uczenia sieci konwolucyjnej PneumoniaCNN: spadek ważonej funkcji straty (lewy wykres) oraz wzrost dokładności Accuracy i pola pod krzywą ROC-AUC na zbiorze walidacyjnym (prawy wykres)."
    )

    add_heading_3(doc, "Dobór hiperparametrów:")
    add_body_paragraph(
        doc,
        "Dla klasycznych modeli uczenia maszynowego przeprowadzono systematyczne przeszukiwanie siatki parametrów "
        "(GridSearchCV) z 5-krotną stratyfikowaną, grupową walidacją krzyżową "
        "(5-fold StratifiedGroupKFold; bez wycieku pacjentów/studiów) na zbiorze treningowym, optymalizując "
        "metrykę ROC-AUC. Preprocessing (skalowanie, PCA, cechy domenowe) znajduje się wewnątrz potoków sklearn. "
        "Zestawienie badanych przestrzeni poszukiwań oraz optymalnych konfiguracji prezentuje poniższa tabela."
    )

    add_formatted_table(
        doc,
        headers=["Model", "Przeszukiwana przestrzeń hiperparametrów", "Wybrana konfiguracja optymalna", "5-Fold CV / Val ROC-AUC"],
        rows=build_hyperparam_rows(summary),
    )

    add_heading_3(doc, "Ewaluacja:")
    add_body_paragraph(
        doc,
        "W diagnostyce medycznej opieranie się wyłącznie na ogólnej dokładności (Accuracy) jest niewystarczające "
        "przy niezbalansowanych klasach. Z tego względu ewaluację oparto na szerokim zestawie metryk klinicznych: "
        "Czułości (Recall / Sensitivity – zdolność wykrycia chorych na zapalenie płuc), Swoistości (Specificity – "
        "zdolność poprawnego wykluczenia choroby u pacjenta zdrowego), Precyzji (PPV – Positive Predictive Value), "
        "zbalansowanej dokładności (Balanced Accuracy), mierze F1-score oraz polach pod krzywymi ROC-AUC i PR-AUC "
        "(Average Precision). Wyniki uzyskane na zbiorze walidacyjnym (n = 524) oraz niezależnym zbiorze testowym "
        "(n = 624 – przy progu 0,50 oraz przy progach ustalonych na walidacji) "
        "zestawiono w tabeli poniżej oraz na Rysunkach 5 i 6. Wartości pochodzą z "
        "`models/metrics_summary.json`."
    )

    add_formatted_table(
        doc,
        headers=["Model (Zbiór i tryb)", "Accuracy", "Balanced Acc", "Precision", "Recall (Czułość)", "Specificity (Swoistość)", "F1-score", "ROC-AUC"],
        rows=build_evaluation_rows(summary),
    )

    paired = summary.get("bootstrap_paired_auc_comparisons_test", [])
    if paired:
        add_heading_3(doc, "Bootstrap: porównania par ROC-AUC na teście")
        pair_rows = []
        for item in paired:
            ci = ""
            if item.get("ci_diff_low") is not None:
                ci = (
                    f"różnica {item['mean_auc_diff']:.4f} "
                    f"[{item['ci_diff_low']:.4f}, {item['ci_diff_high']:.4f}], "
                    f"P(A>B)={item['p_a_greater_b']:.3f}"
                )
            pair_rows.append([item["model_a"], item["model_b"], ci or "—"])
        add_formatted_table(
            doc,
            headers=["Model A", "Model B", "Paired bootstrap (AUC różnica / P(A>B))"],
            rows=pair_rows,
        )

    cnn_key = "PneumoniaCNN (PyTorch)"
    rs = summary["results_summary"]
    val_cnn = rs[cnn_key]["val"]
    test_def = rs[cnn_key]["test_default"]
    test_thr = rs[cnn_key]["test_val_threshold"]
    boot = rs[cnn_key].get("bootstrap_roc_auc", {})
    boot_txt = ""
    if boot.get("ci_low") is not None:
        boot_txt = (
            f" (bootstrap 95% CI: {_pl_auc(boot['ci_low'])}–{_pl_auc(boot['ci_high'])})"
        )
    add_body_paragraph(
        doc,
        f"Na zbiorze walidacyjnym sieć PneumoniaCNN osiąga dokładność {_pl_pct(val_cnn['accuracy'])}, "
        f"czułość {_pl_pct(val_cnn['recall'])} oraz ROC-AUC {_pl_auc(val_cnn['roc_auc'])}. "
        f"Na niezależnym zbiorze testowym (n = 624) ROC-AUC wynosi {_pl_auc(test_def['roc_auc'])}{boot_txt}, "
        f"a PR-AUC {_pl_auc(test_def['pr_auc'])}. Przy progu domyślnym 0,50 sieć uzyskuje czułość "
        f"{_pl_pct(test_def['recall'])} i swoistość {_pl_pct(test_def['specificity'])}. "
        f"Po nałożeniu progu ustalonego wyłącznie na walidacji (Youden J, bez strojenia na teście) "
        f"metryki testowe CNN to m.in. balanced accuracy {_pl_pct(test_thr['balanced_accuracy'])}, "
        f"precyzja {_pl_pct(test_thr['precision'])} i swoistość {_pl_pct(test_thr['specificity'])}."
    )

    add_figure_with_caption(
        doc,
        FIGURES_DIR / "fig5_model_comparison_roc_pr.png",
        "Rysunek 5. Porównanie modeli na zbiorze testowym (n = 624): krzywe ROC (lewo), krzywe Precision-Recall (środek) "
        "oraz metryki przy progach z walidacji (prawo)."
    )

    add_figure_with_caption(
        doc,
        FIGURES_DIR / "fig6_confusion_matrices.png",
        "Rysunek 6. Macierze pomyłek (Confusion Matrices) na zbiorze testowym (n = 624) dla wszystkich czterech badanych modeli."
    )

    # =========================================================================
    # 4. INTERPRETACJA WYNIKÓW I WYJAŚNIALNOŚĆ
    # =========================================================================
    add_heading_1(doc, "4. Interpretacja wyników i wyjaśnialność")

    add_heading_3(doc, "Feature importance / Mapy istotności / Grad-CAM:")
    add_body_paragraph(
        doc,
        "W zastosowaniach medycznych modele uczenia maszynowego nie mogą funkcjonować jako tzw. 'czarne skrzynki' "
        "(black-box), ponieważ lekarz podejmujący decyzję terapeutyczną musi zweryfikować, czy model opiera swoją "
        "predykcję na rzeczywistych zmianach patologicznych w miąższu płucnym, czy na artefaktach pozaanatomicznych. "
        "W projekcie zrealizowano wielopoziomową analizę wyjaśnialności (Rysunek 8):"
    )
    add_bullet_point(
        doc,
        "Przestrzenna istotność pikseli 2D (Random Forest): ",
        "Współczynniki istotności Giniego dla 4 096 cech pikselowych przekształcono z powrotem do wymiaru 64×64 "
        "i nałożono na uśrednioną sylwetkę klatki piersiowej (Rysunek 8, lewy górny panel). Najwyższe wartości "
        "istotności układają się symetrycznie w środkowych i dolnych polach obu płuc (z wyłączeniem cienia kręgosłupa "
        "i mięśni naramiennych), co dowodzi, że las losowy faktycznie analizuje przejrzystość miąższu płucnego."
    )
    add_bullet_point(
        doc,
        "Przestrzenna mapa wag PCA (Regresja Logistyczna): ",
        "Poprzez iloczyn skalarny wektora 100 wag modelu liniowego z macierzą składowych głównych PCA (Eigenlungs) "
        "odtworzono efektywną mapę wag w przestrzeni obrazu (Rysunek 8, środkowy górny panel). Obszary czerwone "
        "(dodatni wpływ na predykcję PNEUMONIA) pokrywają się z centralnymi strefami płucnymi, natomiast obszary "
        "niebieskie odpowiadają zarysom przepony i kątów przeponowo-żebrowych, których ostre odcięcie świadczy "
        "o braku płynu i nacieku (klasa NORMAL)."
    )
    add_bullet_point(
        doc,
        "Ranking cech domenowych: ",
        "Wśród 17 wyekstrahowanych cech inżynierskich najwyższą wagę diagnostyczną w modelu Random Forest uzyskały: "
        "gradient pionowy i poziomy (odzwierciedlające zatarcie rysunku naczyniowo-oskrzelowego i krawędzi żeber przez "
        "wysięk zapalny), odchylenie standardowe jasności (kontrast globalny) oraz wskaźnik asymetrii lewo-prawo "
        "(charakterystyczny dla jednostronnych płatowych zapaleń bakteryjnych)."
    )
    add_bullet_point(
        doc,
        "Mapy aktywacji Grad-CAM dla sieci PneumoniaCNN: ",
        "Dla głębokiej sieci konwolucyjnej zaimplementowano metodę Gradient-weighted Class Activation Mapping "
        "(Grad-CAM) na ostatnim bloku konwolucyjnym (block4 o rozdzielczości 16×16). Jak pokazano w dolnym rzędzie "
        "Rysunku 8, sieć z bardzo wysoką precyzją lokalizuje ogniska konsolidacji zapalnej w prawym oraz lewym "
        "płucu, ignorując znaczniki literowe 'R' oraz kości obojczyków."
    )

    add_figure_with_caption(
        doc,
        FIGURES_DIR / "fig8_explainability_gradcam_importance.png",
        "Rysunek 8. Wyjaśnialność modeli (XAI): dwuwymiarowa mapa istotności pikseli Random Forest (góra-lewo), przestrzenna mapa wag Regresji Logistycznej przez odwrotną transformatę PCA (góra-środek), ranking cech domenowych (góra-prawo) oraz mapy cieplne Grad-CAM sieci PneumoniaCNN na zdjęciach testowych (dół)."
    )

    add_heading_3(doc, "Analiza błędów:")
    add_body_paragraph(
        doc,
        "Szczegółowa analiza przypadków błędnie sklasyfikowanych przez sieć PneumoniaCNN na zbiorze testowym "
        "(Rysunek 9) pozwoliła zidentyfikować główne przyczyny kliniczne i techniczne pomyłek:"
    )
    add_bullet_point(
        doc,
        "Przypadki fałszywie dodatnie (False Positives – FP, zdrowe płuca sklasyfikowane jako PNEUMONIA): ",
        "Analiza obrazów FP (Rysunek 9, dwa pierwsze zdjęcia po lewej) wykazuje, że błędy te występują głównie "
        "na zdjęciach wykonanych w fazie niepełnego wdechu (hipowentylacji), typowej dla płaczących lub "
        "niewspółpracujących małych dzieci. Przy płytkim wdechu kopuły przepony ustawione są wysoko, sylwetka "
        "serca i grasicy ulega pozornemu poszerzeniu, a stłoczenie naczyń płucnych w dolnych polach imituje "
        "zagęszczenia zapalne."
    )
    add_bullet_point(
        doc,
        "Przypadki fałszywie ujemne (False Negatives – FN, nierozpoznane zapalenie płuc): ",
        "Porównanie skuteczności detekcji względem podtypu etiologicznego (Rysunek 9, wykres słupkowy po prawej) "
        "ujawnia kluczową prawidłowość kliniczną: sieć PneumoniaCNN prawidłowo wykrywa aż 90,5% przypadków "
        "bakteryjnego zapalenia płuc (charakteryzującego się wyraźnym naciekiem płatowym), natomiast w przypadku "
        "wirusowego zapalenia płuc skuteczność spada do 84,5% (w trybie przesiewowym przy progu 0,50 skuteczność "
        "przekracza 98,9% dla obu grup). We wczesnej fazie infekcji wirusowej zmiany ograniczają się do delikatnego "
        "pogrubienia ścian oskrzeli i siateczkowatych zacienień okołownękowych, które po przeskalowaniu obrazu do "
        "128×128 px mają bardzo niski kontrast względem prawidłowego miąższu."
    )

    add_figure_with_caption(
        doc,
        FIGURES_DIR / "fig9_error_analysis.png",
        "Rysunek 9. Analiza błędów (Error Analysis) na zbiorze testowym: przykłady błędów fałszywie dodatnich (FP #1, FP #2), fałszywie ujemnych (FN #1, FN #2) oraz porównanie skuteczności rozpoznania względem etiologii (NORMAL vs BACTERIA vs VIRUS)."
    )

    # =========================================================================
    # 5. IMPLEMENTACJA
    # =========================================================================
    add_heading_1(doc, "5. Implementacja")

    add_heading_3(doc, "Interfejs użytkownika lub API:")
    add_body_paragraph(
        doc,
        "W ramach projektu zaimplementowano kompletną, interaktywną aplikację webową wspomagającą diagnostykę "
        "medyczną w oparciu o framework Streamlit (plik app.py). System ładuje wytrenowane potoki klasycznego "
        "uczenia maszynowego (models/classical_models.joblib) oraz wagi sieci konwolucyjnej PyTorch "
        "(models/cnn_pneumonia.pt) i udostępnia intuicyjny interfejs graficzny dla lekarza radiologa lub pediatry."
    )
    add_bullet_point(
        doc,
        "Funkcjonalności aplikacji: ",
        "(1) wybór dowolnego zdjęcia RTG ze zbioru testowego (z filtrowaniem po podtypie NORMAL, BACTERIA, VIRUS) "
        "lub wgranie własnego pliku obrazu w formacie JPEG/PNG; (2) skalibrowane prawdopodobieństwa Platta oraz "
        "osobne, zamrożone progi walidacyjne dla każdego modelu (opcjonalne nadpisanie progu CNN w trybie eksperymentalnym); "
        "(3) natychmiastowe generowanie i nakładanie "
        "mapy cieplnej Grad-CAM wskazującej lokalizację nacieku zapalnego; (4) równoległe zestawienie prawdopodobieństw "
        "P(PNEUMONIA) zwróconych przez wszystkie 4 wytrenowane modele (PneumoniaCNN, SVM, Random Forest, "
        "Logistic Regression) na wspólnym wykresie porównawczym oraz w tabeli diagnostycznej (Rysunek 10)."
    )

    add_figure_with_caption(
        doc,
        FIGURES_DIR / "fig10_ui_preview.png",
        "Rysunek 10. Działanie panelu diagnostycznego aplikacji Streamlit (app.py) dla dwóch reprezentatywnych przypadków: pacjenta z bakteryjnym zapaleniem płuc (górny wiersz) oraz pacjenta zdrowego (dolny wiersz)."
    )

    # =========================================================================
    # 6. PODSUMOWANIE
    # =========================================================================
    add_heading_1(doc, "6. Podsumowanie")

    add_body_paragraph(
        doc,
        "W zrealizowanym projekcie przeprowadzono pełny proces badawczo-inżynieryjny obejmujący eksplorację, "
        "przetwarzanie, modelowanie oraz wyjaśnialną interpretację klasyfikacji zdjęć RTG klatki piersiowej "
        "w kierunku zapalenia płuc na zbiorze Chest X-Ray / PneumoniaMNIST (n = 5 856). Wykazano, że zarówno "
        "odpowiednio zaprojektowane klasyczne potoki uczenia maszynowego (oparte na redukcji wymiarowości PCA "
        "oraz ekstrakcji cech domenowych), jak i głębokie sieci konwolucyjne pozwalają na wysoce skuteczną "
        "automatyzację diagnostyki. Najlepsze wyniki uzyskała autorska sieć konwolucyjna PneumoniaCNN (PyTorch); "
        "metryki liczbowe są generowane z `models/metrics_summary.json` i nie wykorzystują etykiet testowych "
        "do strojenia progów. Połączenie predykcji sieci z wizualizacją "
        "Grad-CAM w aplikacji Streamlit wpisuje się bezpośrednio w potrzeby nowoczesnej praktyki klinicznej jako "
        "narzędzie tzw. szybkiego triażu radiologicznego (wskazujące zdjęcia wymagające pilnej konsultacji lekarskiej).",
        bold_prefix="Wnioski: "
    )

    add_body_paragraph(
        doc,
        "System ma charakter wyłącznie badawczo-edukacyjny i nie jest przeznaczony do zastosowań klinicznych. "
        "Głównym ograniczeniem analizowanego zbioru danych jest jego jednoośrodkowe pochodzenie (pediatryczni "
        "pacjenci w wieku 1–5 lat z jednego szpitala w Kantonie) oraz zauważalne przesunięcie rozkładu (domain shift) "
        "pomiędzy podzbiorem treningowym a testowym. Ponadto kompresja obrazów do rozdzielczości 64×64 lub 128×128 px "
        "powoduje utratę drobnych detali siateczkowatych, co utrudnia wykrywanie wczesnych stadiów wirusowego "
        "zapalenia płuc. Dodatkowym wyzwaniem pozostaje brak informacji o numerze ID pacjenta dla wszystkich podzbiorów "
        "oraz obecność artefaktów medycznych (elektrody, cewniki) mogących korelować z ciężkością stanu klinicznego.",
        bold_prefix="Problemy i ograniczenia: "
    )

    add_body_paragraph(
        doc,
        "W przyszłych pracach system można rozwinąć w kilku kierunkach: (1) rozszerzenie klasyfikacji binarnej "
        "do zadania trójklasowego (NORMAL vs BACTERIA vs VIRUS), co wspomogłoby decyzję o wdrożeniu lub uniknięciu "
        "antybiotykoterapii; (2) zastosowanie wstępnej segmentacji płuc (np. architekturą U-Net) w celu "
        "automatycznego wycięcia obszarów pozapłucnych przed klasyfikacją; (3) wykorzystanie wielkoskalowych "
        "architektur konwolucyjnych i wizyjnych transformerów (DenseNet-121, ConvNeXt, ViT) trenowanych w wyższej "
        "rozdzielczości (224×224 lub 512×512 px) z użyciem Transfer Learningu na zbiorach CheXpert oraz NIH ChestX-ray14.",
        bold_prefix="Możliwości rozwoju: "
    )

    # =========================================================================
    # 7. ZAŁĄCZNIKI
    # =========================================================================
    add_heading_1(doc, "7. Załączniki")
    add_body_paragraph(
        doc,
        "Integralną część projektu stanowi repozytorium kodu źródłowego oraz wygenerowanych artefaktów znajdujące się "
        "w katalogu projektu UM/PROJEKT1:"
    )
    add_bullet_point(doc, "PneumoniaMNIST_Analiza_i_Modelowanie.ipynb – ", "interaktywny notatnik Jupyter zawierający wyniki EDA, inżynierii cech, redukcji wymiarowości PCA, trenowania modeli, ewaluacji oraz wizualizacji XAI / Grad-CAM.")
    add_bullet_point(doc, "run_pipeline.py – ", "kompletny skrypt potoku obliczeniowego pobierający zbiór przez kagglehub, trenujący modele i generujący wykresy (Rysunki 1–10) oraz plik podsumowania metryk models/metrics_summary.json.")
    add_bullet_point(doc, "build_report.py / build_notebook.py – ", "generatory raportu DOCX oraz notatnika Jupyter zsynchronizowanych z models/metrics_summary.json.")
    add_bullet_point(doc, "app.py – ", "aplikacja webowa z interfejsem graficznym (Streamlit) do interaktywnej diagnostyki zdjęć RTG płuc (uruchamiana poleceniem: uv run streamlit run app.py).")
    add_bullet_point(doc, "Katalog figures/ – ", "zestaw 10 wykresów i wizualizacji w wysokiej rozdzielczości (fig1_class_distribution.png – fig10_ui_preview.png).")
    add_bullet_point(doc, "Katalog models/ – ", "zapisane wagi wytrenowanej sieci konwolucyjnej (cnn_pneumonia.pt), potoki modeli klasycznych (classical_models.joblib) oraz metadane zbioru (dataset_metadata.csv).")

    REPORT_DIR = REPORT_PATH.parent
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    out_path_1 = REPORT_PATH.resolve()
    doc.save(str(out_path_1))

    print(f"[SUCCESS] Saved DOCX reports to:\n  - {out_path_1}")

if __name__ == "__main__":
    build_report()
