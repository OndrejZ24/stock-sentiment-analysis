- **Sběr Reddit dat**: Automatické získávání příspěvků a komentářů o akciích
- **Oracle databáze**: Robustní ukládání dat s automatickým znovupřipojením
- **Zpracování textu**: Pokročilé čištění textu a detekce akciových symbolů
- **Příprava pro sentiment analýzu**: Optimalizovaná data pro modely strojového učení
- **Odolnost vůči chybám**: Pokračuje ve zpracování i když chybí závislosti

### Python závislosti
```bash
pip install pandas numpy tqdm oracledb praw python-dotenv
```

### Volitelné závislosti (pro lepší výsledky)
```bash
pip install nltk spacy
python -m spacy download en_core_web_sm
```

## 🛠️ Nastavení

### 1. Stažení a instalace
```bash
git clone <tvoje-repo>
cd stock-sentiment-analysis
pip install pandas numpy tqdm oracledb praw python-dotenv
```

### 2. Konfigurace prostředí
Vytvoř `.env` soubor s tvými přístupovými údaji:
```env
# Oracle Database
db-dsn=tvoje_oracle_connection_string
db-username=tvoje_db_uzivatelske_jmeno
db-password=tvoje_db_heslo

# Reddit API
api-client_id=tvoje_reddit_client_id
api-client_secret=tvoje_reddit_client_secret
api-username=tvoje_reddit_uzivatelske_jmeno
api-password=tvoje_reddit_heslo
```

### 3. Test nastavení
```bash
python test_setup.py
```

## 📁 Struktura projektu

```
stock-sentiment-analysis/
├── utils.py                 # Hlavní utility funkce
├── data-import.ipynb        # Notebook pro sběr Reddit dat
├── preprocessing.ipynb      # Notebook pro čištění a zpracování dat
├── test_setup.py           # Script pro ověření systému
├── us_tickers.csv          # Akciové symboly (automaticky stahované)
├── .env                    # Proměnné prostředí (vytvoř si)
└── README.md               # Tento soubor
```

## 🔧 Jak to používat

### 1. Sběr dat
Spusť `data-import.ipynb` notebook pro:
- Připojení k Reddit API
- Stažení příspěvků a komentářů ze sledovaných subredditů
- Uložení dat do Oracle databáze
- Aktualizaci časových razítek

### 2. Zpracování dat
Spusť `preprocessing.ipynb` notebook pro:
- Načtení surových dat z Oracle
- Čištění a normalizaci textu
- Detekci zmínek akciových symbolů
- Přidání časových a engagement funkcí
- Přípravu dat pro analýzu sentimentu

### 3. Klíčové funkce

#### Detekce akciových symbolů
```python
from utils import detect_tickers_in_text

text = "Kupuju $AAPL a GOOGL teď!"
ticker_set = {"AAPL", "GOOGL", "TSLA"}
tickers = detect_tickers_in_text(text, ticker_set)
# Vrátí: ["AAPL", "GOOGL"]
```

#### Normalizace textu
```python
from utils import normalize_text_for_sentiment

text = "Podívej na https://example.com pro $AAPL info!"
normalized = normalize_text_for_sentiment(text, keep_tickers=True)
# Vrátí: "Podívej na pro $AAPL info"
```

#### Odstranění stop slov (NLTK/spaCy)
```python
from utils import remove_financial_stopwords, remove_stopwords_spacy

text = "Myslím si, že AAPL je super akcie pro buy"
# S NLTK
clean_nltk = remove_financial_stopwords(text, preserve_tickers=True)
# Nebo s spaCy (lepší)
clean_spacy = remove_stopwords_spacy(text, preserve_tickers=True)
```

## 📊 Datové schéma

### Vstupní data (z Redditu)
- **Příspěvky**: title, body, score, created_utc, author, subreddit, url, upvote_ratio
- **Komentáře**: body, score, created_utc, author, parent_post_id, subreddit

### Zpracovaná data na výstupu
- **text**: Původní textový obsah
- **sentiment_ready_text**: Vyčištěný text pro analýzu
- **mentioned_tickers**: Seznam nalezených akciových symbolů
- **n_tickers**: Počet zmíněných akciových symbolů
- **text_length**: Počet znaků
- **word_count**: Počet slov
- **score_log1p**: Log-transformované skóre
- **date, hour, day_of_week**: Časové funkce
- **is_weekend**: Indikátor víkendu

## 🔍 Řešení problémů

### Časté problémy

1. **Chyby importu**
   ```bash
   pip install --upgrade pandas numpy tqdm oracledb praw python-dotenv
   ```

2. **Problémy s databázovým připojením**
   - Zkontroluj přístupové údaje v `.env` souboru
   - Ověř konektivitu k Oracle databázi
   - Ujisti se, že firewall povoluje připojení

3. **Chyby Reddit API**
   - Zkontroluj Reddit API přístupové údaje
   - Zkontroluj rate limity
   - Ujisti se, že user agent je unikátní

4. **Problémy s pamětí**
   - Zmenši `TEST_SAMPLE_SIZE` v preprocessingu
   - Zpracovávej data v menších dávkách
   - Správně zavírej databázová připojení

5. **Pomalé zpracování textu**
   - spaCy je pomalejší, ale přesnější než NLTK
   - Pro větší datasety zvažuj NLTK
   - Použij progress bary pro sledování pokroku

### Kontrola závislostí
```bash
python -c "from utils import check_dependencies; check_dependencies()"
```

## 🏗️ Architektura

### Hlavní komponenty

1. **utils.py**: Hlavní utility knihovna
   - Databázová připojení s auto-retry
   - Zpracování textu a detekce akciových symbolů
   - Orchestrace datového pipeline
   - Graceful handling chybějících závislostí

2. **data-import.ipynb**: Sběr dat
   - Integrace s Reddit API
   - Sledování subredditů
   - Inkrementální aktualizace dat

3. **preprocessing.ipynb**: Zpracování dat
   - Čištění a normalizace textu
   - Feature engineering
   - Detekce a analýza akciových symbolů
   - Profesionální odstranění stop slov

### Zpracování chyb
- Automatické znovupřipojení k databázi
- Graceful degradation při chybějících závislostech
- Komprehensivní logování a reportování chyb
- Retry logika pro síťové operace

## 📈 Výkonnostní úvahy

- **Databáze**: Používá connection pooling a prepared statements
- **Paměť**: Zpracovává data v konfigurovatelných dávkách
- **CPU**: Využívá progress bary pro dlouho běžící operace
- **Síť**: Implementuje retry logiku a rate limiting
- **NLP**: spaCy vs NLTK trade-off (přesnost vs rychlost)

## 🎯 Výsledky

Pipeline vytváří:
- **sentiment_ready_data.csv**: Hlavní dataset pro sentiment analýzu
- **Vyčištěný text**: Odstraněny URLs, stop slova, normalizované
- **Akciové symboly**: Přesná detekce s minimem false positives
- **Časové funkce**: Pro analýzu trendů v čase
- **Engagement metriky**: Score, upvote ratio, délka textu

## 🚧 Další vývoj

### Přidání nových funkcí
1. Přidej funkce do `utils.py`
2. Aktualizuj type hints a dokumentaci
3. Přidej error handling pro chybějící závislosti
4. Otestuj s `test_setup.py`

### Testování
```bash
python test_setup.py  # Ověření systému
python utils.py       # Kontrola závislostí a základní testy
```

## � Tipy

- **Rychlost**: Pro velké datasety použij NLTK místo spaCy
- **Přesnost**: Pro lepší výsledky použij spaCy
- **Paměť**: Zpracovávej data po dávkách
- **Monitoring**: Sleduj progress bary a logy
- **Testování**: Vždy spusť `test_setup.py` před začátkem

## 📞 Podpora

Pro problémy a otázky:
1. Zkontroluj sekci řešení problémů
2. Spusť `python test_setup.py` pro diagnostiku
3. Zkontroluj logy pro detailní chybové zprávy
4. GitHub Issues pro hlášení bugů

---
*Projekt vytvořen pro analýzu sentimentu akciových diskusí na českých a anglických fórech. Letzgoo! 🚀*