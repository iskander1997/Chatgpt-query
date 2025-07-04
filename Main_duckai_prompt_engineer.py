import time
import re
import json
import os
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from openpyxl.styles import Font, Alignment
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('duckduckgo_analyzer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DuckDuckGoAIBot:
    def __init__(self, headless=False, timeout=45):
        self.timeout = timeout
        self.driver = None
        self.wait = None
        self.headless = headless
        self.setup_driver()

    def setup_driver(self):
        """Setup Chrome WebDriver"""
        logger.info("🔧 Setting up Chrome WebDriver...")
        options = Options()
        if self.headless:
            #options.add_argument('--headless')
            logger.info("🔧 Running in headless mode")
        
        # Add additional options for stability
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        try:
            self.driver = webdriver.Chrome(options=options)
            self.wait = WebDriverWait(self.driver, self.timeout)
            logger.info("✅ Chrome WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Chrome WebDriver: {e}")
            raise

    def start_chat(self):
        """Start DuckDuckGo AI chat session"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🌐 Attempting to load DuckDuckGo AI Chat (attempt {attempt + 1})")
                self.driver.get("https://duckduckgo.com/?q=DuckDuckGo+AI+Chat&ia=chat&duckai=1")
                logger.info(f"🌐 Page loaded successfully (attempt {attempt + 1})")

                # Wait for and click "Essayez par vous-même" button
                essayez_button = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Essayez par vous-même')]"))
                )
                essayez_button.click()
                logger.info("✅ 'Essayez par vous-même' button clicked")

                # Wait for and click "Je suis d'accord" button
                agree_button = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Je suis d\'accord")]'))
                )
                agree_button.click()
                logger.info("✅ Agreement button clicked")
                
                # Wait for chat interface to be ready
                time.sleep(3)
                return True
                
            except Exception as e:
                logger.error(f"❌ Error starting chat session (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise

    def send_message(self, prompt):
        """Send message to DuckDuckGo AI and get response"""
        try:
            # Count current responses before sending message
            initial_response_count = len(self.driver.find_elements(By.CSS_SELECTOR, "div.VrBPSncUavA1d7C9kAc5"))
            
            # Find and interact with textarea
            textarea = self.wait.until(
                EC.element_to_be_clickable((By.TAG_NAME, "textarea"))
            )

            logger.info("📝 Entering prompt...")
            textarea.clear()  # Clear any existing text
            textarea.send_keys(prompt)
            time.sleep(1)
            
            logger.info("🔍 Looking for send button...")
            send_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[@type='submit' or contains(@class, 'send') or contains(@aria-label, 'Send')]"))  
            )
            
            logger.info("📤 Sending message...")
            send_button.click()
            logger.info("✅ Send button clicked")

            # Wait for response
            logger.info("⏳ Waiting for AI response...")
            response_timeout = 60  # 60 seconds maximum
            
            for i in range(0, response_timeout, 5):
                try:
                    current_responses = self.driver.find_elements(By.CSS_SELECTOR, "div.VrBPSncUavA1d7C9kAc5")
                    if len(current_responses) > initial_response_count:
                        logger.info(f"🎉 New response detected after {i} seconds")
                        time.sleep(2)  # Wait for initial response to load

                        # Wait for response to stabilize
                        prev_length = 0
                        stable_count = 0
                        
                        for check in range(3):  # 36 seconds max
                            try:
                                current_responses = self.driver.find_elements(By.CSS_SELECTOR, "div.VrBPSncUavA1d7C9kAc5")
                                if current_responses:
                                    current_length = len(current_responses[-1].text)
                                    if current_length == prev_length and current_length > 50:
                                        stable_count += 1
                                        if stable_count >= 3:
                                            logger.info("✅ Response appears complete and stable")
                                            break
                                    else:
                                        stable_count = 0
                                    prev_length = current_length
                                        
                            except Exception as e:
                                logger.warning(f"⚠️ Error checking response stability: {e}")
                                
                            time.sleep(3)
                        
                        break
                    
                    if i % 20 == 0 and i > 0:
                        logger.info(f"⏳ Still waiting for response... ({i}s/{response_timeout}s)")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error checking for response: {e}")
                
                time.sleep(5)
            else:
                logger.error(f"❌ No response received within {response_timeout} seconds")
                return None

            # Extract response text
            logger.info("📖 Extracting response text...")
            try:
                response_divs = self.driver.find_elements(By.CSS_SELECTOR, "div.VrBPSncUavA1d7C9kAc5")
                if response_divs:
                    latest_response = response_divs[-1].text
                    logger.info(f"✅ Response extracted: {len(latest_response)} characters")
                    logger.info(f"📄 Response preview: {latest_response[:200]}...")
                    return latest_response
                else:
                    logger.error("❌ No response elements found")
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Error extracting response: {e}")
                return None

        except Exception as e:
            logger.error(f"❌ Send message failed: {e}")
            return None

    def close(self):
        """Close the browser completely"""
        logger.info("🔒 Closing browser completely...")
        try:
            if self.driver:
                # Close all tabs
                for handle in self.driver.window_handles:
                    try:
                        self.driver.switch_to.window(handle)
                        self.driver.close()
                    except:
                        pass
                
                self.driver.quit()
                logger.info("🔒 Browser closed successfully")
        except Exception as e:
            logger.warning(f"⚠️ Error closing browser: {e}")
        finally:
            self.driver = None
            self.wait = None
            # Wait for processes to close
            time.sleep(2)

    def restart_session(self):
        """Restart complete session - close browser and reopen"""
        logger.info("🔄 Restarting complete session...")
        
        # Close current browser
        self.close()
        
        # Recreate and start new session
        self.setup_driver()
        self.start_chat()
        
        logger.info("✅ Session restarted successfully")
        time.sleep(2)  # Stabilization

class PressReleaseAnalyzer:
    def __init__(self, model="DuckDuckGo-AI", output_dir="analyses", headless=True):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.headless = headless
        self.bot = None
        self.init_bot()

        self.supported_categories = [
            "resultats_financiers",
            "dividendes", 
            "projets_strategiques",
            "changement_direction",
            "partenariat",
            "litiges_reglementations"
        ]

        self._compiled_patterns = {
            'percentage': re.compile(r'([+-]?\d+[,.]?\d*)\s*%'),
            'amount': re.compile(r'(\d+[,.]?\d*)\s*(millions?|milliards?)\s*(?:de\s*)?(dinars?|euros?|dollars?)', re.IGNORECASE)
        }

    def init_bot(self):
        """Initialize bot session"""
        logger.info("🚀 Initializing DuckDuckGo AI session...")
        self.bot = DuckDuckGoAIBot(headless=self.headless)
        self.bot.start_chat()

    def extract_financial_metrics(self, text):
        """Extract financial metrics for 'resultats_financiers' category"""
        metrics = {}
        percentages = self._compiled_patterns['percentage'].findall(text)
        amounts = self._compiled_patterns['amount'].findall(text)

        metrics['percentages'] = []
        for p in percentages:
            try:
                metrics['percentages'].append(float(p.replace(',', '.')))
            except ValueError:
                continue

        metrics['amounts'] = amounts
        return metrics

    def extract_dividend_metrics(self, text):
        """Extract metrics for 'dividendes' category"""
        return {
            'dividend_amounts': [],
            'distribution_rates': [],
            'payment_dates': []
        }

    def extract_strategic_metrics(self, text):
        """Extract metrics for 'projets_strategiques' category"""
        return {
            'investment_amounts': [],
            'project_duration': [],
            'target_markets': []
        }

    def extract_management_metrics(self, text):
        """Extract metrics for 'changement_direction' category"""
        return {
            'names': [],
            'positions': [],
            'effective_dates': []
        }

    def extract_partnership_metrics(self, text):
        """Extract metrics for 'partenariat' category"""
        return {
            'partners': [],
            'partnership_amounts': [],
            'sectors': []
        }

    def extract_legal_metrics(self, text):
        """Extract metrics for 'litiges_reglementations' category"""
        return {
            'fine_amounts': [],
            'authorities': [],
            'case_dates': []
        }

    def build_financial_prompt(self, texte_presse, include_metrics=True):
        """Build prompt for 'resultats_financiers' category"""
        base_prompt = f"""Tu es un analyste financier senior spécialisé dans l'évaluation de l'impact boursier des résultats financiers. Réponds UNIQUEMENT en français.

Voici un communiqué de résultats financiers :
\"\"\"{texte_presse}\"\"\"

IMPORTANT : Effectue des recherches web pour contextualiser ton analyse (secteur, concurrents, attentes marché).

Ta mission : Attribue une note de sentiment financier comprise entre -1 et +1 (avec deux décimales) qui reflète l'impact potentiel sur le cours de l'action.

Structure ta réponse ainsi :
1. ANALYSE DES CRITÈRES : Évalue chaque critère et attribue-lui un impact (très négatif, négatif, neutre, positif, très positif)
2. MÉTHODOLOGIE : Explique comment tu pondères chaque élément pour calculer la note finale
3. JUSTIFICATION : Détaille les éléments déterminants qui ont orienté ta notation
4. Termine par : note=[valeur] (ex: note=0.45)"""

        if include_metrics:
            metrics = self.extract_financial_metrics(texte_presse)
            if metrics['percentages']:
                base_prompt += f"\n\nNote : {len(metrics['percentages'])} indicateurs chiffrés détectés dans le texte."
        
        return base_prompt

    def build_dividend_prompt(self, texte_presse, include_metrics=True):
        """Build prompt for 'dividendes' category"""
        return f"""Tu es un analyste spécialisé dans l'évaluation de l'impact boursier des annonces de dividendes. Réponds UNIQUEMENT en français.

Voici un communiqué sur les dividendes :
\"\"\"{texte_presse}\"\"\"

IMPORTANT : Effectue des recherches web pour contextualiser (historique dividendes, pratiques sectorielles, taux actuels).

Ta mission : Attribue une note de sentiment comprise entre -1 et +1 (avec deux décimales) qui reflète l'impact sur l'attractivité de l'action pour les investisseurs.

Structure ta réponse ainsi :
1. ANALYSE DES CRITÈRES : Évalue chaque critère et attribue-lui un impact (très négatif, négatif, neutre, positif, très positif)
2. MÉTHODOLOGIE : Explique comment tu pondères chaque élément pour calculer la note finale
3. JUSTIFICATION : Détaille les éléments déterminants qui ont orienté ta notation
4. Termine par : note=[valeur] (ex: note=0.62)"""

    def build_strategic_prompt(self, texte_presse, include_metrics=True):
        """Build prompt for 'projets_strategiques' category"""
        return f"""Tu es un analyste stratégique spécialisé dans l'évaluation de l'impact boursier des projets d'entreprise. Réponds UNIQUEMENT en français.

Voici un communiqué sur un projet stratégique :
\"\"\"{texte_presse}\"\"\"

IMPORTANT : Effectue des recherches web pour contextualiser (tendances secteur, projets similaires, positionnement concurrentiel).

Ta mission : Attribue une note de sentiment comprise entre -1 et +1 (avec deux décimales) qui reflète le potentiel de création de valeur du projet.

Structure ta réponse ainsi :
1. ANALYSE DES CRITÈRES : Évalue chaque critère et attribue-lui un impact (très négatif, négatif, neutre, positif, très positif)
2. MÉTHODOLOGIE : Explique comment tu pondères chaque élément pour calculer la note finale
3. JUSTIFICATION : Détaille les éléments déterminants qui ont orienté ta notation
4. Termine par : note=[valeur] (ex: note=0.78)"""

    def build_management_prompt(self, texte_presse, include_metrics=True):
        """Build prompt for 'changement_direction' category"""
        return f"""Tu es un analyste spécialisé dans l'évaluation de l'impact boursier des changements de direction. Réponds UNIQUEMENT en français.

Voici un communiqué sur un changement de direction :
\"\"\"{texte_presse}\"\"\"

IMPORTANT : Effectue des recherches web pour contextualiser (profil des dirigeants, historique entreprise, réactions marché).

Ta mission : Attribue une note de sentiment comprise entre -1 et +1 (avec deux décimales) qui reflète l'impact du changement sur la confiance des investisseurs.

Structure ta réponse ainsi :
1. ANALYSE DES CRITÈRES : Évalue chaque critère et attribue-lui un impact (très négatif, négatif, neutre, positif, très positif)
2. MÉTHODOLOGIE : Explique comment tu pondères chaque élément pour calculer la note finale
3. JUSTIFICATION : Détaille les éléments déterminants qui ont orienté ta notation
4. Termine par : note=[valeur] (ex: note=-0.25)"""

    def build_partnership_prompt(self, texte_presse, include_metrics=True):
        """Build prompt for 'partenariat' category"""
        return f"""Tu es un analyste spécialisé dans l'évaluation de l'impact boursier des partenariats stratégiques. Réponds UNIQUEMENT en français.

Voici un communiqué sur un partenariat :
\"\"\"{texte_presse}\"\"\"

IMPORTANT : Effectue des recherches web pour contextualiser (profil des partenaires, synergies sectorielles, deals similaires).

Ta mission : Attribue une note de sentiment comprise entre -1 et +1 (avec deux décimales) qui reflète le potentiel de création de valeur du partenariat.

Structure ta réponse ainsi :
1. ANALYSE DES CRITÈRES : Évalue chaque critère et attribue-lui un impact (très négatif, négatif, neutre, positif, très positif)
2. MÉTHODOLOGIE : Explique comment tu pondères chaque élément pour calculer la note finale
3. JUSTIFICATION : Détaille les éléments déterminants qui ont orienté ta notation
4. Termine par : note=[valeur] (ex: note=0.55)"""

    def build_legal_prompt(self, texte_presse, include_metrics=True):
        """Build prompt for 'litiges_reglementations' category"""
        return f"""Tu es un analyste spécialisé dans l'évaluation de l'impact boursier des questions juridiques et réglementaires. Réponds UNIQUEMENT en français.

Voici un communiqué sur des litiges ou réglementations :
\"\"\"{texte_presse}\"\"\"

IMPORTANT : Effectue des recherches web pour contextualiser (précédents juridiques, impact secteur, évolutions réglementaires).

Ta mission : Attribue une note de sentiment comprise entre -1 et +1 (avec deux décimales) qui reflète l'impact sur la valorisation et les risques futurs.

Structure ta réponse ainsi :
1. ANALYSE DES CRITÈRES : Évalue chaque critère et attribue-lui un impact (très négatif, négatif, neutre, positif, très positif)
2. MÉTHODOLOGIE : Explique comment tu pondères chaque élément pour calculer la note finale
3. JUSTIFICATION : Détaille les éléments déterminants qui ont orienté ta notation
4. Termine par : note=[valeur] (ex: note=-0.45)"""

    def get_category_methods(self, category):
        """Return extraction and prompt methods for a given category"""
        category_mapping = {
            "resultats_financiers": {
                "extract": self.extract_financial_metrics,
                "prompt": self.build_financial_prompt
            },
            "dividendes": {
                "extract": self.extract_dividend_metrics,
                "prompt": self.build_dividend_prompt
            },
            "projets_strategiques": {
                "extract": self.extract_strategic_metrics,
                "prompt": self.build_strategic_prompt
            },
            "changement_direction": {
                "extract": self.extract_management_metrics,
                "prompt": self.build_management_prompt
            },
            "partenariat": {
                "extract": self.extract_partnership_metrics,
                "prompt": self.build_partnership_prompt
            },
            "litiges_reglementations": {
                "extract": self.extract_legal_metrics,
                "prompt": self.build_legal_prompt
            }
        }

        if category not in category_mapping:
            raise ValueError(f"Catégorie '{category}' non supportée. Catégories disponibles: {self.supported_categories}")

        return category_mapping[category]

    def analyze_press_release(self, text, category, include_metrics=True):
        """Analyze a press release with automatic restart after each response"""
        logger.info(f"🔍 Analyse de la catégorie: {category}")

        methods = self.get_category_methods(category)
        extract_func = methods["extract"]
        prompt_func = methods["prompt"]

        metrics = extract_func(text)
        logger.info(f"📊 Métriques extraites pour '{category}': {len(str(metrics))} caractères")

        prompt = prompt_func(text, include_metrics)

        # Attempt analysis with retry on failure
        for attempt in range(2):  # 2 attempts maximum
            try:
                logger.info(f"🤖 Tentative d'analyse {attempt + 1}/2")
                start_time = time.time()
                
                # Send message
                analysis = self.bot.send_message(prompt)
                processing_time = time.time() - start_time
                
                if analysis and analysis.strip() and len(analysis) > 100:
                    # Check that it's not a duplication of the prompt
                    if not any(keyword in analysis for keyword in ["Tu es un analyste", "Ta mission", "Structure ta réponse"]):
                        logger.info(f"✅ Analyse réussie en {processing_time:.2f} secondes")
                        
                        # AUTOMATIC RESTART AFTER EACH SUCCESSFUL RESPONSE
                        logger.info("🔄 Restart automatique de la session...")
                        try:
                            self.bot.restart_session()
                            logger.info("✅ Session redémarrée avec succès")
                        except Exception as restart_error:
                            logger.warning(f"⚠️ Erreur lors du restart: {restart_error}")
                            # In case of restart failure, recreate the bot completely
                            logger.info("🔄 Recréation complète du bot...")
                            self.bot.close()
                            self.init_bot()
                        
                        return {
                            "category": category,
                            "metrics": metrics,
                            "analysis": analysis,
                            "processing_time": processing_time
                        }
                    else:
                        logger.warning("⚠️ Détection de duplication de prompt dans la réponse")
                        raise Exception("Réponse contient des éléments du prompt")
                else:
                    raise Exception("Réponse invalide, vide ou trop courte")
                    
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'analyse (tentative {attempt + 1}): {e}")
                
                if attempt < 1:  # If not the last attempt
                    logger.info("🔄 Restart de la session suite à l'erreur...")
                    try:
                        self.bot.restart_session()
                        logger.info("✅ Session redémarrée après erreur")
                    except Exception as restart_error:
                        logger.error(f"❌ Échec du restart: {restart_error}")
                        # Recreate bot completely
                        logger.info("🔄 Recréation complète du bot...")
                        self.bot.close()
                        self.init_bot()
                else:
                    # Last attempt failed - restart anyway for next analysis
                    logger.info("🔄 Restart final après échec...")
                    try:
                        self.bot.restart_session()
                    except Exception:
                        logger.error("❌ Échec du restart final")
                        self.bot.close()
                        self.init_bot()

        # If all attempts failed
        return {
            "category": category,
            "metrics": metrics,
            "analysis": "ERREUR: Échec après 2 tentatives",
            "processing_time": 0
        }

    def save_analysis_optimized(self, text, category, analysis, processing_time, metrics=None, tokens_used=None):
        """Save analysis with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{category}_{timestamp}.json"
        filepath = self.output_dir / filename

        text_length = len(text)
        word_count = len(text.split())

        data = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "model": self.model,
            "processing_time_seconds": processing_time,
            "input_text": text,
            "analysis": analysis,
            "text_length": text_length,
            "word_count": word_count,
            "detected_metrics": metrics,
            "tokens_used": tokens_used
        }

        try:
            with open(filepath, 'w', encoding='utf-8', buffering=8192) as f:
                json.dump(data, f, ensure_ascii=False, indent=2, separators=(',', ':'))
            logger.info(f"💾 Analyse sauvegardée : {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"❌ Erreur de sauvegarde: {str(e)}")
            return None

    def save_intermediate_results(self, results, output_file):
        """Save intermediate results"""
        try:
            backup_file = output_file.replace('.xlsx', f'_backup_{datetime.now().strftime("%H%M%S")}.xlsx')
            results_df = pd.DataFrame(results)
            
            with pd.ExcelWriter(backup_file, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Analyses', index=False)
                
            logger.info(f"💾 Sauvegarde intermédiaire: {backup_file}")
            return backup_file
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde intermédiaire: {str(e)}")
            return None

    def close(self):
        """Close DuckDuckGo AI session and browser"""
        logger.info("🔒 Fermeture de la session DuckDuckGo AI...")
        if self.bot:
            self.bot.close()

def main():
    parser = argparse.ArgumentParser(description="Analyseur de communiqués de presse avec DuckDuckGo AI")
    parser.add_argument("--category", choices=["resultats_financiers", "dividendes", "projets_strategiques", 
                                             "changement_direction", "partenariat", "litiges_reglementations"],
                       default="resultats_financiers", help="Catégorie du communiqué")
    parser.add_argument("--input-file", default="fichier1.xlsx", help="Fichier Excel d'entrée")
    parser.add_argument("--output-file", default="output.xlsx", help="Fichier Excel de sortie")
    parser.add_argument("--sheet-name", default="feuil1", help="Nom de la feuille Excel d'entrée")
    parser.add_argument("--headless", action="store_true", help="Mode sans interface graphique")
    args = parser.parse_args()

    analyzer = None
    
    try:
        analyzer = PressReleaseAnalyzer(headless=args.headless)

        print("🏦 ANALYSEUR DE COMMUNIQUÉS DE PRESSE - DUCKDUCKGO AI")
        
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Le fichier {args.input_file} n'existe pas")

        print(f"📖 Lecture du fichier {args.input_file}...")
        try:
            df = pd.read_excel(args.input_file, sheet_name=args.sheet_name)
        except Exception as e:
            raise Exception(f"Erreur lors de la lecture du fichier Excel: {e}")

        if 'extracted_text' not in df.columns:
            print("❌ La colonne 'extracted_text' n'existe pas dans le fichier Excel")
            print(f"colonnes disponibles: {list(df.columns)}")
            user_column = input("Entrer le nom de la colonne contenant le texte à analyser: ")
            if user_column not in df.columns:
                raise ValueError("La colonne 'extracted_text' n'existe pas dans le fichier Excel")
            else:
                df["extracted_text"] = df[user_column]

        print(f"📊 {len(df)} lignes trouvées dans le fichier")

        results = []

        for index, row in df.iterrows():
            ligne_numero = index + 2  
            texte = str(row['extracted_text']) if pd.notna(row['extracted_text']) else ""

            print(f"\n{'='*50}")
            print(f"🔍 TRAITEMENT LIGNE {ligne_numero}")
            print(f"{'='*50}")

            if not texte.strip():
                print(f"⚠️  Ligne {ligne_numero}: Texte vide, passage à la suivante")
                results.append({
                    'extracted_text': texte,
                    'analyse': "ERREUR: Texte vide",
                    'note': "Ligne ignorée - texte vide"
                })
                continue

            print(f"📝 Longueur du texte: {len(texte)} caractères")
            print(f"🔤 Nombre de mots: {len(texte.split())} mots")
            print(f"📋 Catégorie: {args.category}")

            try:
                print("🤖 Analyse en cours avec DuckDuckGo AI...")
                result = analyzer.analyze_press_release(texte, args.category)

                if result["analysis"] and not result["analysis"].startswith("ERREUR"):
                    print("✅ Analyse réussie!")
                    print(f"⏱️  Temps de traitement: {result['processing_time']:.2f} secondes")

                    analyse_preview = result["analysis"][:1000] + "..." if len(result["analysis"]) > 200 else result["analysis"]
                    print(f"📄 Aperçu de l'analyse: {analyse_preview}")

                    results.append({
                        'extracted_text': texte,
                        'analyse': result["analysis"],
                        'note': f"Succès - {result['processing_time']:.2f}s - {result['metrics']}"
                    })

                    # Sauvegarde individuelle de l'analyse
                    analyzer.save_analysis_optimized(texte, args.category, result["analysis"], 
                                                   result["processing_time"], result["metrics"])

                else:
                    print("❌ Échec de l'analyse")
                    results.append({
                        'extracted_text': texte,
                        'analyse': "ERREUR: Analyse échouée",
                        'note': "Erreur lors de l'analyse"
                    })

                if ligne_numero < len(df):  # Pas de fermeture après la dernière ligne
                    print("🔄 FERMETURE DE LA SESSION ACTUELLE...")
                    try:
                        analyzer.close()  # Ferme complètement la session, y compris le navigateur
                        print("✅ Session fermée")
                        
                        print("🚀 CRÉATION D'UN NOUVEL ANALYZER...")
                        analyzer = PressReleaseAnalyzer(headless=args.headless)
                        print("✅ Nouvel analyzer créé avec succès")
                        
                        # Pause pour stabilisation
                        time.sleep(2)
                        
                    except Exception as restart_error:
                        print(f"❌ Erreur lors de la recréation: {restart_error}")
                        print("⚠️ Arrêt du traitement")
                        break
                    
                # Sauvegarde intermédiaire tous les 5 éléments
                if len(results) % 5 == 0:
                    try:
                        analyzer.save_intermediate_results(results, args.output_file)
                    except Exception:
                        print("❌ Erreur lors de la sauvegarde intermédiaire")

                print(f"📈 Progression: {ligne_numero}/{len(df)} lignes traitées")
            except Exception as e:
                print(f"❌ Exception lors du traitement de la ligne {ligne_numero}: {e}")
                results.append({
                    'extracted_text': texte,
                    'analyse': f"ERREUR: {str(e)}",
                    'note': "Exception lors du traitement"
                })

        print(f"\n{'='*50}")
        print("💾 SAUVEGARDE DES RÉSULTATS FINAUX")
        print(f"{'='*50}")

        try:
            results_df = pd.DataFrame(results)

            with pd.ExcelWriter(args.output_file, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Analyses', index=False)

                workbook = writer.book
                worksheet = writer.sheets['Analyses']

                worksheet.column_dimensions['A'].width = 50  
                worksheet.column_dimensions['B'].width = 80  
                worksheet.column_dimensions['C'].width = 30  

                header_font = Font(bold=True)
                for cell in worksheet[1]:
                    cell.font = header_font

                for row in worksheet.iter_rows(min_row=2, max_col=3):
                    row[0].alignment = Alignment(wrap_text=True, vertical='top')  
                    row[1].alignment = Alignment(wrap_text=True, vertical='top')  
                    row[2].alignment = Alignment(wrap_text=True, vertical='top')  

            print(f"✅ Résultats sauvegardés dans: {args.output_file}")

            successful_analyses = len([r for r in results if not r['analyse'].startswith('ERREUR')])
            total_analyses = len(results)

            print(f"\n📊 STATISTIQUES FINALES")
            print(f"📈 Analyses réussies: {successful_analyses}/{total_analyses}")
            print(f"📉 Analyses échouées: {total_analyses - successful_analyses}/{total_analyses}")
            print(f"📋 Taux de réussite: {(successful_analyses/total_analyses)*100:.1f}%")
            print(f"⏰ Traitement terminé: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            raise

    except Exception as e:
        logger.error(f"❌ Erreur générale: {e}")
        print(f"\n❌ Erreur: {e}")

    finally:
        if analyzer:
            analyzer.close()

if __name__ == "__main__":
    main()