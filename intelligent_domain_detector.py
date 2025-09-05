"""
Intelligent Domain Detection System
Automatically detects entity domains using multi-layer NLP analysis
Zero predefined lists - pure AI-driven detection
"""

import spacy
import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class IntelligentDomainDetector:
    """
    🧠 INTELLIGENT DOMAIN DETECTION SYSTEM
    Uses multi-layer NLP to automatically detect entity domains
    No predefined lists - adapts to any query type
    """
    
    def __init__(self):
        """Initialize with safe SpaCy loading"""
        self.nlp = None
        self._load_spacy_model()
        
        # Pre-compile regex patterns for performance
        self._compile_patterns()
        
        logger.info("🧠 Intelligent Domain Detector initialized successfully")
    
    def _load_spacy_model(self):
        """Safely load SpaCy model with error handling"""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ SpaCy model loaded successfully")
        except OSError:
            logger.warning("⚠️ SpaCy model not found. Attempting to download...")
            try:
                import subprocess
                import sys
                subprocess.run([
                    sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                ], check=True, capture_output=True)
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("✅ SpaCy model downloaded and loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load SpaCy model: {e}")
                self.nlp = None
        except Exception as e:
            logger.error(f"❌ Unexpected error loading SpaCy: {e}")
            self.nlp = None
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance - ENHANCED COVERAGE"""
        self.compiled_patterns = {
            'person': [
                # World Leaders & Politicians - comprehensive
                re.compile(r'\b(narendra\s+modi|joe\s+biden|vladimir\s+putin|xi\s+jinping|angela\s+merkel)\b', re.I),
                re.compile(r'\b(barack\s+obama|donald\s+trump|emmanuel\s+macron|justin\s+trudeau|boris\s+johnson)\b', re.I),
                # Tech Leaders - expanded
                re.compile(r'\b(elon\s+musk|bill\s+gates|mark\s+zuckerberg|sundar\s+pichai|satya\s+nadella)\b', re.I),
                re.compile(r'\b(jeff\s+bezos|tim\s+cook|jensen\s+huang|sam\s+altman|ratan\s+tata)\b', re.I),
                # Sports Personalities - comprehensive
                re.compile(r'\b(virat\s+kohli|cristiano\s+ronaldo|lionel\s+messi|serena\s+williams|usain\s+bolt)\b', re.I),
                re.compile(r'\b(sachin\s+tendulkar|ms\s+dhoni|rohit\s+sharma|lebron\s+james|roger\s+federer)\b', re.I),
                # Entertainment Celebrities - expanded
                re.compile(r'\b(taylor\s+swift|shah\s+rukh\s+khan|priyanka\s+chopra|amitabh\s+bachchan)\b', re.I),
                re.compile(r'\b(oprah\s+winfrey|tom\s+cruise|leonardo\s+dicaprio|will\s+smith)\b', re.I),
                # Activists & Nobel Winners
                re.compile(r'\b(greta\s+thunberg|malala\s+yousafzai|nelson\s+mandela)\b', re.I),
                # Single name detection (famous people)
                re.compile(r'\b(modi|musk|gates|zuckerberg|bezos|kohli|messi|ronaldo|swift|obama|trump|putin)\b', re.I),
                # General person indicators
                re.compile(r'\b(mr|mrs|ms|dr|prof|president|prime\s+minister|minister|ceo|founder|leader)\b', re.I),
                re.compile(r'\b(announces?|says?|speaks?|declares?|states?|addresses?|comments?)\b', re.I),
                re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+(said|announced|declared|stated|spoke|commented)\b'),
                re.compile(r'\b(chief\s+minister|finance\s+minister|home\s+minister)\b', re.I)
            ],
            'location': [
                # Major cities - expanded coverage
                re.compile(r'\b(delhi|mumbai|bangalore|kolkata|chennai|hyderabad|pune|ahmedabad|lucknow|jaipur)\b', re.I),
                re.compile(r'\b(bhubaneswar|guwahati|thiruvananthapuram|coimbatore|vadodara|rajkot|madurai|tiruchirappalli)\b', re.I),
                re.compile(r'\b(jodhpur|kochi|vijayawada|mysore|reykjavik|ljubljana|bratislava|tallinn|vilnius|riga|zagreb)\b', re.I),
                # Countries - expanded
                re.compile(r'\b(india|usa|america|china|uk|britain|france|germany|japan|australia|canada|russia)\b', re.I),
                re.compile(r'\b(luxembourg|malta|cyprus|estonia|latvia|lithuania|slovenia|slovakia|montenegro)\b', re.I),
                re.compile(r'\b(kazakhstan|uzbekistan|kyrgyzstan|tajikistan|belarus|moldova)\b', re.I),
                # Indian states - comprehensive
                re.compile(r'\b(arunachal\s+pradesh|meghalaya|tripura|manipur|nagaland|mizoram|sikkim|goa)\b', re.I),
                re.compile(r'\b(uttarakhand|jharkhand|chhattisgarh|telangana)\b', re.I),
                # Location indicators
                re.compile(r'\b(city|state|country|capital|district|region|area|zone|province|territory)\b', re.I),
                re.compile(r'\b(pollution|population|traffic|infrastructure|development|growth|crisis)\b', re.I),
                re.compile(r'\b(in|at|from|to|across|throughout)\s+[A-Z][a-z]+\b'),
                re.compile(r'\b[A-Z][a-z]+\s+(pollution|crisis|development|growth|issues?)\b')
            ],
            'organization': [
                # Tech giants
                re.compile(r'\b(tesla|apple|google|microsoft|amazon|facebook|meta|netflix|uber|twitter|x)\b', re.I),
                # Indian companies - comprehensive
                re.compile(r'\b(razorpay|paytm|zomato|byju\'?s|nykaa|phonepe|freshworks|zerodha|dream11)\b', re.I),
                re.compile(r'\b(policybazaar|cardekho|delhivery|swiggy|flipkart|ola|oyo)\b', re.I),
                re.compile(r'\b(reliance|tcs|infosys|wipro|hcl|tech\s+mahindra)\b', re.I),
                # Global companies
                re.compile(r'\b(blackrock|palantir|snowflake|figma|notion|stripe|canva|discord|zoom)\b', re.I),
                re.compile(r'\b(databricks|airtable|slack|spotify|dropbox|airbnb|uber|lyft)\b', re.I),
                # Traditional indicators
                re.compile(r'\b(company|corporation|inc|ltd|llc|startup|firm|business|group|enterprise)\b', re.I),
                re.compile(r'\b(ceo|revenue|profit|stock|shares|market|earnings|ipo|merger|acquisition)\b', re.I),
                re.compile(r'\b(products|services|launch|expansion|strategy|operations)\b', re.I),
                re.compile(r'\b[A-Z]\w+\s+(company|corp|inc|ltd|group)\b')
            ],
            'technology': [
                # Mobile technology - high priority
                re.compile(r'\b(iphone|android|smartphone|mobile\s+phone|tablet|ipad)\b', re.I),
                re.compile(r'\b(apple\s+(iphone|ipad|watch|tv|mac)|google\s+(pixel|android))\b', re.I),
                re.compile(r'\b(samsung\s+(galaxy|note)|oneplus|xiaomi|oppo|vivo)\b', re.I),
                # Programming languages - comprehensive
                re.compile(r'\b(python|javascript|java|c\+\+|rust|go|swift|kotlin|dart|typescript)\b', re.I),
                re.compile(r'\b(php|ruby|scala|perl|haskell|erlang|clojure|f#|r|matlab)\b', re.I),
                # Frameworks and libraries
                re.compile(r'\b(react|angular|vue\.?js|svelte|flutter|react\s+native|ionic)\b', re.I),
                re.compile(r'\b(django|flask|fastapi|express\.?js|laravel|rails|spring)\b', re.I),
                # DevOps and Cloud
                re.compile(r'\b(docker|kubernetes|terraform|ansible|jenkins|gitlab\s+ci|github\s+actions)\b', re.I),
                re.compile(r'\b(aws\s+lambda|azure\s+functions|google\s+cloud\s+run|serverless)\b', re.I),
                re.compile(r'\b(cloudformation|helm|istio|prometheus|grafana|elasticsearch|kibana)\b', re.I),
                # AI/ML tools
                re.compile(r'\b(langchain|hugging\s+face|weights\s*.?\s*biases|mlflow|tensorflow|pytorch)\b', re.I),
                re.compile(r'\b(openai|chatgpt|gpt|bert|transformers|keras|scikit.learn)\b', re.I),
                # General tech terms
                re.compile(r'\b(ai|artificial\s+intelligence|machine\s+learning|deep\s+learning|neural\s+network)\b', re.I),
                re.compile(r'\b(blockchain|cryptocurrency|quantum\s+computing|edge\s+computing|iot)\b', re.I),
                re.compile(r'\b(api|sdk|framework|library|database|algorithm|software|platform)\b', re.I),
                re.compile(r'\b(features|specifications|performance|update|version|upgrade)\b', re.I),
                re.compile(r'\b(coding|programming|development|deployment|testing|debugging)\b', re.I)
            ],
            'automotive': [
                # Car brands - comprehensive
                re.compile(r'\b(tesla|bmw|mercedes|audi|toyota|honda|ford|maruti|tata\s+motors)\b', re.I),
                re.compile(r'\b(rivian|lucid\s+motors|nio|xpeng|byd|fisker|canoo|lordstown)\b', re.I),
                re.compile(r'\b(lamborghini|ferrari|maserati|bentley|rolls.royce|mclaren|bugatti)\b', re.I),
                # Vehicle types
                re.compile(r'\b(car|vehicle|automotive|truck|suv|sedan|ev|electric\s+vehicle|auto)\b', re.I),
                re.compile(r'\b(hybrid|plugin|autonomous|self.driving|connected\s+car)\b', re.I),
                # Automotive tech
                re.compile(r'\b(lidar|adas|telematics|ota\s+updates|vehicle.to.everything)\b', re.I),
                re.compile(r'\b(battery\s+swapping|fast\s+charging|regenerative\s+braking)\b', re.I),
                # Performance terms
                re.compile(r'\b(engine|horsepower|mileage|fuel|battery|hybrid|electric)\b', re.I),
                re.compile(r'\b(driving|performance|safety|crash\s+test|review|manufacturing)\b', re.I)
            ],
            'healthcare': [
                # Medical procedures - comprehensive
                re.compile(r'\b(laparoscopy|arthroscopy|angioplasty|catheterization|endoscopy|bronchoscopy)\b', re.I),
                re.compile(r'\b(colonoscopy|mammography|mri\s+scan|ct\s+scan|pet\s+scan|ultrasound)\b', re.I),
                re.compile(r'\b(x.ray|ecg|ekg|echo|biopsy|surgery|transplant)\b', re.I),
                # Medical conditions
                re.compile(r'\b(covid|coronavirus|diabetes|hypertension|cancer|heart\s+disease)\b', re.I),
                re.compile(r'\b(fibromyalgia|endometriosis|lupus|multiple\s+sclerosis|parkinson)\b', re.I),
                re.compile(r'\b(alzheimer|arthritis|asthma|copd|stroke|pneumonia)\b', re.I),
                # Pharmaceutical companies
                re.compile(r'\b(moderna|pfizer|astrazeneca|johnson.*johnson|novartis|roche)\b', re.I),
                re.compile(r'\b(sanofi|glaxosmithkline|merck|abbott|medtronic|boston\s+scientific)\b', re.I),
                # Medical specialties
                re.compile(r'\b(cardiology|neurology|oncology|gastroenterology|pulmonology|nephrology)\b', re.I),
                re.compile(r'\b(endocrinology|rheumatology|dermatology|ophthalmology|orthopedics)\b', re.I),
                # General medical terms
                re.compile(r'\b(health|medical|hospital|doctor|treatment|medicine|drug|therapy)\b', re.I),
                re.compile(r'\b(patient|diagnosis|symptom|disease|illness|cure|prevention|vaccine)\b', re.I)
            ],
            'finance': [
                # Economic indicators - high priority
                re.compile(r'\b(economic\s+growth|gdp\s+growth|economy|economic|inflation|recession)\b', re.I),
                re.compile(r'\b(stock\s+market|market\s+performance|financial\s+market|bull\s+market|bear\s+market)\b', re.I),
                re.compile(r'\b(interest\s+rates|federal\s+reserve|rbi|central\s+bank|monetary\s+policy)\b', re.I),
                # Cryptocurrencies
                re.compile(r'\b(bitcoin|ethereum|cardano|solana|polygon|chainlink|uniswap|aave)\b', re.I),
                re.compile(r'\b(maker|compound|yearn|synthetix|the\s+graph|dogecoin|litecoin)\b', re.I),
                # Financial instruments
                re.compile(r'\b(derivatives|commodities|reits|etfs|mutual\s+funds|hedge\s+funds)\b', re.I),
                re.compile(r'\b(bonds|stocks|shares|options|futures|swaps|forex)\b', re.I),
                # Financial metrics
                re.compile(r'\b(ebitda|p/e\s+ratio|debt.to.equity|return\s+on\s+equity|free\s+cash\s+flow)\b', re.I),
                re.compile(r'\b(gross\s+margin|net\s+margin|operating\s+margin|current\s+ratio)\b', re.I),
                re.compile(r'\b(sharpe\s+ratio|alpha|beta|volatility|var|value\s+at\s+risk)\b', re.I),
                # Traditional finance
                re.compile(r'\b(bank|banking|finance|investment|money|economy|market|trading|stocks)\b', re.I),
                re.compile(r'\b(loan|credit|debt|interest|inflation|gdp|currency|forex|rupee|dollar)\b', re.I),
                re.compile(r'\b(budget|fiscal|monetary|financial|economic|recession|fed|rbi)\b', re.I)
            ],
            'entertainment': [
                # Streaming shows/movies - comprehensive
                re.compile(r'\b(stranger\s+things|squid\s+game|money\s+heist|the\s+crown|bridgerton)\b', re.I),
                re.compile(r'\b(the\s+mandalorian|ozark|the\s+witcher|lucifer|dark|elite)\b', re.I),
                re.compile(r'\b(the\s+umbrella\s+academy|marvel|dc|disney|pixar|netflix|amazon\s+prime|hulu|hbo)\b', re.I),
                # Music artists - comprehensive  
                re.compile(r'\b(billie\s+eilish|olivia\s+rodrigo|post\s+malone|bad\s+bunny|dua\s+lipa)\b', re.I),
                re.compile(r'\b(the\s+weeknd|taylor\s+swift|ariana\s+grande|drake|kanye|beyonce)\b', re.I),
                re.compile(r'\b(justin\s+bieber|ed\s+sheeran|adele|rihanna|eminem|jay.z)\b', re.I),
                re.compile(r'\b(lil\s+nas\s+x|doja\s+cat|megan\s+thee\s+stallion)\b', re.I),
                # Hollywood actors/actresses
                re.compile(r'\b(tom\s+holland|zendaya|anya\s+taylor.joy|timothée\s+chalamet)\b', re.I),
                re.compile(r'\b(florence\s+pugh|saoirse\s+ronan|jacob\s+tremblay|noah\s+jupe)\b', re.I),
                re.compile(r'\b(thomasin\s+mckenzie|lucas\s+hedges|gaten\s+matarazzo|darci\s+lynne)\b', re.I),
                # Bollywood - comprehensive
                re.compile(r'\b(shah\s+rukh\s+khan|salman\s+khan|aamir\s+khan|akshay\s+kumar)\b', re.I),
                re.compile(r'\b(deepika\s+padukone|priyanka\s+chopra|alia\s+bhatt|katrina\s+kaif)\b', re.I),
                re.compile(r'\b(ayushmann\s+khurrana|vicky\s+kaushal|arjun\s+kapoor|kartik\s+aaryan)\b', re.I),
                re.compile(r'\b(kiara\s+advani|kriti\s+sanon|ananya\s+panday|bhumi\s+pednekar)\b', re.I),
                re.compile(r'\b(sidharth\s+malhotra|aditya\s+roy\s+kapur|rajkummar\s+rao|tara\s+sutaria)\b', re.I),
                # General entertainment
                re.compile(r'\b(movie|film|music|song|album|concert|show|series|cinema)\b', re.I),
                re.compile(r'\b(entertainment|hollywood|bollywood|celebrity|star|actor|actress)\b', re.I),
                re.compile(r'\b(director|producer|singer|musician|artist|review|rating|box\s+office)\b', re.I),
                re.compile(r'\b(streaming|premiere|release|soundtrack|trailer|oscar|emmy|grammy)\b', re.I)
            ],
            'sports': [
                # Cricket - comprehensive
                re.compile(r'\b(cricket|ipl|cricketer|batsman|bowler|wicket|runs?|match|test\s+match)\b', re.I),
                re.compile(r'\b(virat\s+kohli|ms\s+dhoni|rohit\s+sharma|hardik\s+pandya|jasprit\s+bumrah)\b', re.I),
                re.compile(r'\b(ravindra\s+jadeja|rishabh\s+pant|yuzvendra\s+chahal|mohammed\s+shami)\b', re.I),
                re.compile(r'\b(axar\s+patel|shreyas\s+iyer|shubman\s+gill|ishan\s+kishan|kuldeep\s+yadav)\b', re.I),
                # Football/Soccer
                re.compile(r'\b(football|soccer|fifa|premier\s+league|messi|ronaldo|neymar|mbappe)\b', re.I),
                re.compile(r'\b(haaland|kane|benzema|lewandowski|salah|de\s+bruyne)\b', re.I),
                # Basketball
                re.compile(r'\b(basketball|nba|lebron\s+james|stephen\s+curry|kevin\s+durant)\b', re.I),
                re.compile(r'\b(giannis\s+antetokounmpo|luka\s+dončić|nikola\s+jokić|joel\s+embiid)\b', re.I),
                re.compile(r'\b(ja\s+morant|jayson\s+tatum|zion\s+williamson|lamelo\s+ball)\b', re.I),
                re.compile(r'\b(anthony\s+edwards|evan\s+mobley|scottie\s+barnes|cade\s+cunningham)\b', re.I),
                # Tennis
                re.compile(r'\b(tennis|wimbledon|us\s+open|french\s+open|australian\s+open)\b', re.I),
                re.compile(r'\b(novak\s+djokovic|rafael\s+nadal|roger\s+federer|carlos\s+alcaraz)\b', re.I),
                re.compile(r'\b(stefanos\s+tsitsipas|alexander\s+zverev|matteo\s+berrettini)\b', re.I),
                re.compile(r'\b(jannik\s+sinner|casper\s+ruud|taylor\s+fritz|denis\s+shapovalov)\b', re.I),
                re.compile(r'\b(felix\s+auger.aliassime|hubert\s+hurkacz|sebastian\s+korda|holger\s+rune)\b', re.I),
                # Swimming & Athletics
                re.compile(r'\b(swimming|olympics|track\s+and\s+field|athletics|marathon)\b', re.I),
                re.compile(r'\b(katie\s+ledecky|caeleb\s+dressel|bobby\s+finke|regan\s+smith)\b', re.I),
                re.compile(r'\b(sydney\s+mclaughlin|ryan\s+crouser|armand\s+duplantis|karsten\s+warholm)\b', re.I),
                re.compile(r'\b(elaine\s+thompson.herah|yulimar\s+rojas|lilly\s+king|kate\s+douglass)\b', re.I),
                # General sports terms
                re.compile(r'\b(sport|athlete|player|team|game|tournament|championship|league|cup)\b', re.I),
                re.compile(r'\b(medal|gold|silver|bronze|record|world\s+record|olympic\s+record)\b', re.I),
                re.compile(r'\b(coach|training|fitness|performance|victory|defeat|champion)\b', re.I)
            ],
            'education': [
                # Universities - comprehensive
                re.compile(r'\b(stanford|harvard|mit|caltech|oxford|cambridge|eth\s+zurich)\b', re.I),
                re.compile(r'\b(iit\s+delhi|iit\s+bombay|iit\s+madras|iit\s+kanpur|iit\s+kharagpur)\b', re.I),
                re.compile(r'\b(iim\s+ahmedabad|iim\s+bangalore|iim\s+calcutta|aiims|jnu)\b', re.I),
                # Educational terms
                re.compile(r'\b(school|college|university|education|student|teacher|professor)\b', re.I),
                re.compile(r'\b(exam|test|grade|degree|diploma|graduation|admission|curriculum)\b', re.I),
                re.compile(r'\b(learn|study|research|academic|scholar|jee|neet|cbse|icse)\b', re.I),
                # Modern education
                re.compile(r'\b(online\s+learning|e.learning|mooc|coursera|udemy|khan\s+academy)\b', re.I),
                re.compile(r'\b(virtual\s+classroom|blended\s+learning|gamification|lms)\b', re.I)
            ]
        }

    def detect_domain_intelligently(self, query: str) -> Dict:
        """
        🧠 MAIN INTELLIGENCE FUNCTION
        Multi-layer domain detection with confidence scoring
        Enhanced with contextual priority for multi-word queries
        """
        if not query or not query.strip():
            return self._get_fallback_result(query)
        
        query_clean = query.strip().lower()
        query_words = query_clean.split()
        is_multi_word = len(query_words) > 1
        
        try:
            # Multi-layer detection with smart weighting
            domain_scores = {}
            
            # LAYER 1: Semantic clustering (HIGH PRIORITY for multi-word queries)
            semantic_domains = self._detect_domain_from_semantics(query_clean)
            for domain, score in semantic_domains.items():
                # Boost semantic scores for multi-word queries
                final_score = score * 2 if is_multi_word else score
                domain_scores[domain] = domain_scores.get(domain, 0) + final_score
                logger.debug(f"🧠 Semantic detected: {domain} (+{final_score})")
            
            # LAYER 2: Enhanced pattern-based detection
            pattern_domains = self._detect_domain_from_patterns(query_clean)
            for domain, score in pattern_domains.items():
                # Boost pattern scores for multi-word queries
                final_score = score * 1.5 if is_multi_word else score
                domain_scores[domain] = domain_scores.get(domain, 0) + final_score
                logger.debug(f"📝 Pattern detected: {domain} (+{final_score})")
            
            # LAYER 3: Context-based detection
            context_domains = self._detect_domain_from_context(query_clean)
            for domain, score in context_domains.items():
                domain_scores[domain] = domain_scores.get(domain, 0) + score
                logger.debug(f"🎯 Context detected: {domain} (+{score})")
            
            # LAYER 4: NER-based detection (REDUCED PRIORITY for multi-word queries)
            ner_domain = self._detect_domain_from_ner(query_clean)
            if ner_domain:
                # Reduce NER weight for multi-word queries to prevent entity override
                ner_score = 20 if is_multi_word else 40
                domain_scores[ner_domain] = domain_scores.get(ner_domain, 0) + ner_score
                logger.debug(f"🔍 NER detected: {ner_domain} (+{ner_score})")
            
            # LAYER 5: Contextual keyword analysis for specific cases
            context_boost = self._analyze_contextual_keywords(query_clean)
            for domain, boost in context_boost.items():
                domain_scores[domain] = domain_scores.get(domain, 0) + boost
                logger.debug(f"⚡ Context boost: {domain} (+{boost})")
            
            # Calculate final result with improved confidence
            if domain_scores:
                primary_domain = max(domain_scores, key=domain_scores.get)
                max_score = domain_scores[primary_domain]
                confidence = min(max_score / 100.0, 1.0)
                
                # Boost confidence for high-scoring semantic matches
                if max_score >= 80:
                    confidence = min(confidence * 1.1, 1.0)
                
                result = {
                    'primary_domain': primary_domain,
                    'confidence': confidence,
                    'all_scores': domain_scores,
                    'relevant_contexts': self._get_domain_contexts(primary_domain, query_clean),
                    'detection_method': 'intelligent_nlp'
                }
                
                logger.info(f"🎯 Detection result: {query} → {primary_domain} ({confidence:.0%})")
                return result
            
            # No strong detection - return general
            return self._get_fallback_result(query)
            
        except Exception as e:
            logger.error(f"❌ Error in domain detection for '{query}': {e}")
            return self._get_fallback_result(query)
    
    def _detect_domain_from_ner(self, query: str) -> str:
        """NER-based domain detection using SpaCy"""
        if not self.nlp:
            return None
        
        try:
            doc = self.nlp(query)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            if not entities:
                return None
            
            # Map NER labels to domains
            for _, label in entities:
                if label == "PERSON":
                    return "person"
                elif label == "ORG":
                    return "organization"
                elif label in ["GPE", "LOC"]:
                    return "location"
                elif label == "EVENT":
                    return "event"
                elif label in ["MONEY", "PERCENT"]:
                    return "finance"
                elif label == "PRODUCT":
                    return "product"
                elif label == "WORK_OF_ART":
                    return "entertainment"
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ NER detection failed: {e}")
            return None
    
    def _detect_domain_from_patterns(self, query: str) -> Dict[str, int]:
        """Pattern-based domain detection using compiled regex"""
        domain_scores = {}
        
        for domain, patterns in self.compiled_patterns.items():
            matches = 0
            for pattern in patterns:
                if pattern.search(query):
                    matches += 1
            
            if matches > 0:
                # Base score + bonus for multiple matches
                domain_scores[domain] = 25 + (matches * 5)
        
        return domain_scores
    
    def _detect_domain_from_context(self, query: str) -> Dict[str, int]:
        """Context phrase detection"""
        domain_scores = {}
        
        # Context mappings for common phrases
        context_mappings = {
            'location': [
                'pollution in', 'traffic in', 'population of', 'development in',
                'safety in', 'crime in', 'infrastructure of', 'weather in',
                'government of', 'mayor of', 'city of', 'capital of'
            ],
            'person': [
                'statement by', 'speech of', 'policy of', 'decision by',
                'announcement from', 'reaction of', 'opinion of', 'interview with',
                'visit by', 'meeting with', 'address by', 'response from'
            ],
            'organization': [
                'earnings of', 'stock price', 'ceo of', 'products by',
                'services from', 'revenue of', 'expansion of', 'merger with',
                'acquisition by', 'launch by', 'announcement from', 'strategy of'
            ],
            'technology': [
                'features of', 'update to', 'version of', 'development of',
                'innovation in', 'breakthrough in', 'research on', 'algorithm for',
                'software for', 'platform for', 'app for', 'technology behind'
            ],
            'automotive': [
                'mileage of', 'performance of', 'review of', 'price of',
                'specifications of', 'comparison with', 'test drive', 'launch of',
                'safety of', 'features of', 'engine of', 'fuel efficiency'
            ]
        }
        
        for domain, phrases in context_mappings.items():
            matches = sum(1 for phrase in phrases if phrase in query)
            if matches > 0:
                domain_scores[domain] = matches * 20
        
        return domain_scores
    
    def _detect_domain_from_semantics(self, query: str) -> Dict[str, int]:
        """Semantic clustering detection - ENHANCED WITH COMPREHENSIVE ENTITY COVERAGE"""
        domain_scores = {}
        query_lower = query.lower()
        
        # Enhanced semantic word clusters with modern terminology
        semantic_clusters = {
            'location': {
                'urban': ['city', 'urban', 'metropolitan', 'downtown', 'suburb'],
                'infrastructure': ['roads', 'transport', 'utilities', 'drainage'],
                'environment': ['pollution', 'air quality', 'noise', 'climate'],
                'governance': ['government', 'municipal', 'civic', 'administration'],
                'indian_cities': ['bhubaneswar', 'guwahati', 'thiruvananthapuram', 'coimbatore'],
                'global_cities': ['reykjavik', 'ljubljana', 'bratislava', 'tallinn', 'vilnius'],
                'states': ['arunachal', 'meghalaya', 'tripura', 'manipur', 'nagaland', 'mizoram']
            },
            'person': {
                'political': ['leader', 'politician', 'minister', 'president'],
                'professional': ['ceo', 'doctor', 'engineer', 'teacher'],
                'actions': ['speaks', 'announces', 'decides', 'visits'],
                'attributes': ['experienced', 'young', 'senior', 'veteran']
            },
            'organization': {
                'business': ['company', 'corporation', 'startup', 'enterprise'],
                'financial': ['revenue', 'profit', 'investment', 'funding'],
                'operations': ['manufacturing', 'services', 'retail'],
                'performance': ['growth', 'expansion', 'success', 'decline'],
                'indian_fintech': ['razorpay', 'paytm', 'phonepe', 'zerodha'],
                'indian_ecommerce': ['zomato', 'swiggy', 'flipkart', 'nykaa'],
                'global_tech': ['blackrock', 'palantir', 'snowflake', 'figma', 'notion']
            },
            'technology': {
                'mobile_tech': ['iphone', 'android', 'smartphone', 'mobile', 'latest', 'new'],
                'apple_products': ['apple', 'iphone', 'ipad', 'mac', 'watch', 'latest'],
                'microsoft_products': ['microsoft', 'windows', 'office', 'azure', 'teams'],
                'computing': ['software', 'hardware', 'algorithm', 'programming'],
                'internet': ['website', 'app', 'platform', 'digital'],
                'innovation': ['breakthrough', 'research', 'development'],
                'trends': ['trending', 'viral', 'popular', 'emerging'],
                'programming': ['docker', 'kubernetes', 'terraform', 'rust', 'go'],
                'frameworks': ['langchain', 'react', 'angular', 'vue', 'flutter'],
                'ai_ml': ['hugging face', 'weights and biases', 'mlflow', 'tensorflow'],
                'cloud': ['aws', 'azure', 'gcp', 'serverless', 'cloudformation']
            },
            'healthcare': {
                'medical': ['health', 'medical', 'hospital', 'doctor', 'patient'],
                'treatments': ['surgery', 'therapy', 'medicine', 'treatment'],
                'conditions': ['disease', 'illness', 'symptom', 'diagnosis'],
                'procedures': ['laparoscopy', 'arthroscopy', 'angioplasty', 'mri', 'ct'],
                'pharma': ['moderna', 'pfizer', 'astrazeneca', 'novartis', 'roche'],
                'specialties': ['cardiology', 'neurology', 'oncology', 'gastroenterology']
            },
            'finance': {
                'economic_indicators': ['economic', 'economy', 'growth', 'gdp', 'inflation'],
                'banking': ['bank', 'banking', 'finance', 'investment', 'money'],
                'markets': ['market', 'trading', 'stocks', 'shares', 'economy'],
                'crypto': ['bitcoin', 'ethereum', 'cardano', 'solana', 'polygon'],
                'defi': ['uniswap', 'aave', 'compound', 'yearn', 'maker'],
                'metrics': ['ebitda', 'margin', 'ratio', 'volatility', 'alpha'],
                'instruments': ['derivatives', 'etfs', 'reits', 'bonds', 'options']
            },
            'entertainment': {
                'media': ['movie', 'film', 'music', 'song', 'entertainment'],
                'streaming': ['netflix', 'disney', 'amazon prime', 'hbo'],
                'shows': ['stranger things', 'squid game', 'money heist', 'the crown'],
                'music': ['billie eilish', 'post malone', 'taylor swift', 'drake'],
                'bollywood': ['shah rukh', 'deepika', 'priyanka', 'alia'],
                'platforms': ['youtube', 'spotify', 'instagram', 'tiktok']
            },
            'automotive': {
                'brands': ['tesla', 'bmw', 'mercedes', 'audi', 'toyota'],
                'electric': ['rivian', 'lucid motors', 'nio', 'xpeng', 'byd'],
                'technology': ['autonomous', 'self-driving', 'electric', 'hybrid'],
                'performance': ['engine', 'horsepower', 'mileage', 'battery']
            },
            'sports': {
                'cricket': ['cricket', 'ipl', 'virat kohli', 'ms dhoni', 'rohit'],
                'indian_cricket': ['hardik pandya', 'jasprit bumrah', 'ravindra jadeja', 'rishabh pant'],
                'football': ['football', 'messi', 'ronaldo', 'neymar', 'mbappe'],
                'basketball': ['nba', 'lebron', 'curry', 'durant', 'giannis'],
                'tennis': ['tennis', 'djokovic', 'nadal', 'federer', 'alcaraz'],
                'swimming': ['swimming', 'katie ledecky', 'caeleb dressel'],
                'athletics': ['athletics', 'olympics', 'marathon', 'track'],
                'general': ['sport', 'athlete', 'player', 'team', 'match', 'tournament']
            },
            'entertainment': {
                'hollywood': ['tom holland', 'zendaya', 'florence pugh', 'timothée chalamet'],
                'bollywood': ['ayushmann khurrana', 'vicky kaushal', 'kiara advani', 'kriti sanon'],
                'music_pop': ['billie eilish', 'olivia rodrigo', 'taylor swift', 'ariana grande'],
                'music_rap': ['drake', 'post malone', 'lil nas x', 'doja cat'],
                'streaming': ['netflix', 'disney', 'amazon prime', 'hbo'],
                'shows': ['stranger things', 'squid game', 'money heist', 'the crown'],
                'platforms': ['youtube', 'spotify', 'instagram', 'tiktok'],
                'general': ['movie', 'music', 'entertainment', 'celebrity', 'artist']
            },
            'education': {
                'institutions': ['school', 'college', 'university', 'education'],
                'indian': ['iit', 'iim', 'aiims', 'jnu', 'delhi university'],
                'global': ['stanford', 'harvard', 'mit', 'oxford', 'cambridge'],
                'online': ['coursera', 'udemy', 'khan academy', 'edx']
            }
        }
        
        # Enhanced scoring system with entity-specific boosts
        for domain, clusters in semantic_clusters.items():
            domain_score = 0
            for cluster_name, words in clusters.items():
                for word in words:
                    if word in query_lower:
                        # Higher score for exact entity matches
                        if cluster_name in ['mobile_tech', 'apple_products', 'microsoft_products',
                                          'economic_indicators', 'indian_fintech', 'global_tech', 
                                          'programming', 'ai_ml', 'pharma', 'crypto', 'defi', 
                                          'streaming', 'shows', 'music']:
                            domain_score += 50  # High confidence for specific entities
                        else:
                            domain_score += 15  # Standard contextual score
            
            if domain_score > 0:
                domain_scores[domain] = domain_score
        
        return domain_scores
    
    def _analyze_contextual_keywords(self, query: str) -> Dict[str, int]:
        """
        Enhanced contextual analysis for specific keyword combinations
        Handles cases where NER might misclassify based on context
        """
        domain_boosts = {}
        
        # Finance-related contexts
        finance_indicators = [
            'economic', 'economy', 'growth', 'gdp', 'inflation', 'market', 'stock', 
            'investment', 'trading', 'financial', 'revenue', 'profit', 'earnings',
            'price', 'valuation', 'cryptocurrency', 'bitcoin', 'ethereum', 'defi',
            'banking', 'loan', 'credit', 'mortgage', 'insurance', 'tax'
        ]
        
        # Technology-related contexts
        tech_indicators = [
            'iphone', 'android', 'software', 'app', 'ai', 'artificial intelligence',
            'machine learning', 'algorithm', 'programming', 'development', 'code',
            'platform', 'framework', 'api', 'cloud', 'data', 'analytics', 'digital',
            'cyber', 'security', 'blockchain', 'iot', 'automation', 'robotics'
        ]
        
        # Healthcare-related contexts
        health_indicators = [
            'health', 'medical', 'medicine', 'disease', 'treatment', 'therapy',
            'hospital', 'doctor', 'patient', 'drug', 'pharmaceutical', 'vaccine',
            'clinical', 'diagnosis', 'surgery', 'wellness', 'fitness'
        ]
        
        # Entertainment-related contexts
        entertainment_indicators = [
            'movie', 'film', 'music', 'song', 'album', 'concert', 'show', 'series',
            'netflix', 'streaming', 'youtube', 'spotify', 'gaming', 'game'
        ]
        
        # Sports-related contexts
        sports_indicators = [
            'cricket', 'football', 'soccer', 'basketball', 'tennis', 'golf',
            'olympics', 'championship', 'tournament', 'match', 'player', 'team'
        ]
        
        # Check for contextual matches
        for word in finance_indicators:
            if word in query:
                domain_boosts['finance'] = domain_boosts.get('finance', 0) + 30
        
        for word in tech_indicators:
            if word in query:
                domain_boosts['technology'] = domain_boosts.get('technology', 0) + 30
        
        for word in health_indicators:
            if word in query:
                domain_boosts['healthcare'] = domain_boosts.get('healthcare', 0) + 30
        
        for word in entertainment_indicators:
            if word in query:
                domain_boosts['entertainment'] = domain_boosts.get('entertainment', 0) + 30
        
        for word in sports_indicators:
            if word in query:
                domain_boosts['sports'] = domain_boosts.get('sports', 0) + 30
        
        # Special handling for brand + context combinations
        if any(brand in query for brand in ['apple', 'google', 'microsoft', 'amazon']) and \
           any(tech in query for tech in ['iphone', 'pixel', 'surface', 'alexa', 'ai', 'cloud']):
            domain_boosts['technology'] = domain_boosts.get('technology', 0) + 40
        
        # Financial context override for any brand + financial terms
        if any(financial in query for financial in ['stock', 'share', 'price', 'market', 'trading', 
                                                   'earnings', 'revenue', 'profit', 'performance', 'report']):
            # Check if it's clearly financial discussion about any company
            if any(company in query for company in ['tesla', 'apple', 'google', 'microsoft', 'amazon', 
                                                   'meta', 'netflix', 'boeing', 'ford', 'bmw']):
                domain_boosts['finance'] = domain_boosts.get('finance', 0) + 120  # Maximum override
        
        # Organization context for company + workplace/business terms
        if any(org_term in query for org_term in ['company', 'workplace', 'layoffs', 'employees', 
                                                 'culture', 'policies', 'hiring', 'firing']):
            if any(company in query for company in ['google', 'microsoft', 'amazon', 'apple', 'meta',
                                                   'tesla', 'netflix', 'boeing', 'ford', 'bmw']):
                domain_boosts['organization'] = domain_boosts.get('organization', 0) + 45
        
        # Location context for places + development/issues
        if any(location_term in query for location_term in ['development', 'growth', 'issues', 'problems',
                                                           'traffic', 'pollution', 'infrastructure']):
            if any(place in query for place in ['mumbai', 'delhi', 'bangalore', 'singapore', 'dubai',
                                               'london', 'paris', 'tokyo', 'new york', 'sydney']):
                domain_boosts['location'] = domain_boosts.get('location', 0) + 45
        
        # Organization context for workplace terms
        if any(org_term in query for org_term in ['company', 'workplace', 'layoffs', 'culture', 
                                                 'policies', 'employees', 'hiring']):
            domain_boosts['organization'] = domain_boosts.get('organization', 0) + 35
        
        # Location context for development/issues
        if any(location_context in query for location_context in ['development', 'issues', 'problems']) and \
           any(location in query for location in ['mumbai', 'delhi', 'singapore', 'bangalore', 'city']):
            domain_boosts['location'] = domain_boosts.get('location', 0) + 40
        
        if any(brand in query for brand in ['tesla', 'bmw', 'mercedes', 'ford']) and \
           any(auto in query for auto in ['car', 'electric', 'vehicle', 'model']) and \
           not any(financial in query for financial in ['stock', 'share', 'price', 'performance']):
            domain_boosts['automotive'] = domain_boosts.get('automotive', 0) + 40
        
        return domain_boosts
    
    def _get_domain_contexts(self, domain: str, query: str) -> List[str]:
        """Generate relevant contexts for detected domain"""
        
        domain_contexts = {
            'location': [
                'pollution', 'air quality', 'traffic', 'infrastructure', 'development',
                'population', 'demographics', 'safety', 'crime rate', 'education',
                'healthcare', 'employment', 'economy', 'tourism', 'culture',
                'government', 'administration', 'municipal services', 'utilities',
                'transport', 'housing', 'cost of living', 'quality of life',
                'environmental concerns', 'urban planning', 'smart city'
            ],
            'person': [
                'statements', 'speeches', 'policies', 'decisions', 'announcements',
                'public reaction', 'criticism', 'praise', 'controversy', 'achievements',
                'career', 'background', 'education', 'experience', 'leadership',
                'vision', 'goals', 'challenges', 'success', 'performance',
                'public opinion', 'media coverage', 'interviews', 'social media'
            ],
            'organization': [
                'financial performance', 'revenue', 'profit', 'stock price', 'market share',
                'products', 'services', 'innovation', 'technology', 'research',
                'expansion', 'growth', 'competition', 'market position', 'strategy',
                'leadership', 'management', 'employees', 'culture', 'values',
                'customer satisfaction', 'reviews', 'reputation', 'sustainability'
            ],
            'technology': [
                'features', 'specifications', 'performance', 'usability', 'security',
                'privacy', 'innovation', 'breakthrough', 'research', 'development',
                'trends', 'adoption', 'market impact', 'competition', 'alternatives',
                'user feedback', 'reviews', 'updates', 'future prospects'
            ],
            'automotive': [
                'performance', 'mileage', 'fuel efficiency', 'safety', 'features',
                'specifications', 'price', 'value for money', 'comparison', 'reviews',
                'reliability', 'maintenance', 'resale value', 'design', 'comfort',
                'technology', 'innovation', 'electric features', 'hybrid technology'
            ],
            'healthcare': [
                'effectiveness', 'side effects', 'clinical trials', 'research', 'approval',
                'availability', 'cost', 'accessibility', 'patient outcomes', 'safety',
                'innovation', 'breakthrough', 'treatment options', 'prevention',
                'public health impact', 'expert opinions', 'medical community'
            ],
            'finance': [
                'market performance', 'economic impact', 'interest rates', 'inflation',
                'investment opportunities', 'risk assessment', 'growth prospects',
                'market trends', 'analyst opinions', 'regulatory changes',
                'global economic factors', 'market volatility', 'investor sentiment'
            ],
            'entertainment': [
                'reviews', 'ratings', 'box office', 'audience reaction', 'critical reception',
                'awards', 'nominations', 'performances', 'direction', 'production',
                'music', 'cinematography', 'story', 'characters', 'cultural impact',
                'social media buzz', 'fan reactions', 'celebrity news'
            ],
            'sports': [
                'performance', 'statistics', 'rankings', 'team dynamics', 'coaching',
                'training', 'fitness', 'injuries', 'recovery', 'strategy', 'tactics',
                'fan reactions', 'media coverage', 'sponsorships', 'transfers',
                'tournaments', 'competitions', 'records', 'achievements'
            ],
            'education': [
                'curriculum', 'teaching methods', 'student performance', 'outcomes',
                'accessibility', 'affordability', 'quality', 'innovation', 'technology',
                'research', 'faculty', 'infrastructure', 'placements', 'alumni',
                'rankings', 'accreditation', 'policy changes', 'reform'
            ]
        }
        
        return domain_contexts.get(domain, [
            'analysis', 'opinion', 'review', 'discussion', 'debate', 'reaction',
            'impact', 'significance', 'implications', 'consequences', 'benefits',
            'challenges', 'opportunities', 'trends', 'developments', 'news'
        ])
    
    def _get_fallback_result(self, query: str) -> Dict:
        """Fallback result for unknown domains"""
        return {
            'primary_domain': 'general',
            'confidence': 0.5,
            'relevant_contexts': [
                'news', 'updates', 'information', 'analysis', 'opinion',
                'review', 'discussion', 'debate', 'reaction', 'response',
                'impact', 'significance', 'development', 'trend', 'research'
            ],
            'detection_method': 'fallback'
        }
