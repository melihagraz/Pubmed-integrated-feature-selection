import streamlit as st
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import io
import requests
import xml.etree.ElementTree as ET
import logging

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ---- Set up logging ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# ---- Enhanced PubMed Integration with Article Details ----
class SimplePubMedSearcher:
    """
    Enhanced PubMed searcher with rate limiting and article fetching.
    Uses E-utilities API for searching and fetching article details.

    """
    
    def __init__(self, email: str, api_key: Optional[str] = None, delay: float = 0.34):
        """
        Initializes the PubMed searcher.
        Args:
            email (str): A valid email address for NCBI API usage.
            api_key (Optional[str]): An optional NCBI API key for higher rate limits.
            delay (float): Time in seconds to wait between API requests.
        """
        self.email = email
        self.api_key = api_key
        self.delay = delay if not api_key else 0.1  # Faster with API key
        self.last_request = 0
        self.cache = {}
        self.article_cache = {}  # Cache for article details
    
    def _rate_limit(self):
        """Apply rate limiting between requests"""
        time_since_last = time.time() - self.last_request
        if time_since_last < self.delay:
            time.sleep(self.delay - time_since_last)
        self.last_request = time.time()
    
    def fetch_article_details(self, pmids: list[str]) -> list[dict]:
        
        """
        Fetch PubMed article details for a list of PMIDs.
        """
        articles = []
        if not pmids:
            return articles

        self._rate_limit()

        try:
            # Fetch article details using efetch
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "email": self.email,
                "tool": "FeatureSelectionAgent",
            }
            if self.api_key:
                params["api_key"] = self.api_key

            resp = requests.get(fetch_url, params=params, timeout=15)
            resp.raise_for_status()

            # İçerik gerçekten XML mi?
            ctype = resp.headers.get("Content-Type", "")
            if "xml" not in ctype.lower():
                logging.error(f"efetch non-XML: {resp.text[:200]}")
                st.warning("PubMed efetch non-XML döndü (rate limit olabilir). Tekrar deniyorum...")
                time.sleep(1.0)
                resp = requests.get(fetch_url, params=params, timeout=15)
                resp.raise_for_status()

            # XML parse et
            root = ET.fromstring(resp.content)
            for article_elem in root.findall(".//PubmedArticle"):
                try:
                    article = self._parse_article(article_elem)
                    if article:
                        articles.append(article)
                except Exception as inner_e:
                    logging.error(f"Parse error: {inner_e}")

        except Exception as e:
            logging.error(f"efetch error: {e}")
            st.warning(f"PubMed efetch başarısız: {e}")

        return articles


    
    def _parse_article(self, article_elem) -> dict:
        """Parses a single article XML element into a dictionary."""
        try:
            # Extract PMID
            pmid_elem = article_elem.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else "Unknown"
            
            # Extract title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else "No title"
            
            # Extract authors
            authors = []
            for author in article_elem.findall('.//Author'):
                last_name = author.find('LastName')
                fore_name = author.find('ForeName')
                if last_name is not None:
                    author_name = last_name.text
                    if fore_name is not None:
                        author_name = f"{author_name} {fore_name.text[0]}"
                    authors.append(author_name)
            authors_str = ", ".join(authors[:3])
            if len(authors) > 3:
                authors_str += " et al."
            
            # Extract journal
            journal_elem = article_elem.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else "Unknown Journal"
            
            # Extract year
            year_elem = article_elem.find('.//PubDate/Year')
            year = year_elem.text if year_elem is not None else "Unknown"
            
            # Extract abstract
            abstract_elem = article_elem.find('.//AbstractText')
            abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
            if len(abstract) > 300:
                abstract = abstract[:297] + "..."
            
            return {
                'pmid': pmid,
                'title': title,
                'authors': authors_str,
                'journal': journal,
                'year': year,
                'abstract': abstract,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
        except Exception:
            return None
    
    def search_simple(self, feature_name: str, disease_context: str = None, max_results: int = 5) -> dict:
        """
        Performs a simple PubMed search and calculates an evidence score.
        Args:
            feature_name (str): The name of the feature to search for.
            disease_context (str, optional): A specific disease context.
            max_results (int): Maximum number of results to fetch.
        Returns:
            dict: A dictionary with search results, including a calculated evidence score.
        """
        cache_key = f"{feature_name}_{disease_context}_{max_results}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Rate limiting
        self._rate_limit()
        
        try:
            # Build search query
            query_parts = [f'"{feature_name}"']
            if disease_context:
                query_parts.append(f'"{disease_context}"')
            query_parts.extend(["biomarker", "expression", "association"])
            search_term = " AND ".join(query_parts)
            
            # PubMed E-utilities search
# 1) Build and call ESEARCH with JSON
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": search_term,
                "retmax": max_results,
                "retmode": "json",             # <-- use JSON to avoid XML parsing issues
                "email": self.email,
                "tool": "FeatureSelectionAgent",
                "sort": "relevance",
            }
            if self.api_key:
                params["api_key"] = self.api_key

            resp = requests.get(search_url, params=params, timeout=10)
            resp.raise_for_status()

            # Handle rate limiting explicitly
            if resp.status_code == 429:
                retry = int(resp.headers.get("Retry-After", "1"))
                time.sleep(retry)
                resp = requests.get(search_url, params=params, timeout=10)
                resp.raise_for_status()

            # Parse JSON safely
            data = resp.json()
            ids = data.get("esearchresult", {}).get("idlist", []) or []
            count = int(data.get("esearchresult", {}).get("count", "0"))

            # Calculate evidence score (0-5 scale)
            evidence_score = min(count / 20.0, 5.0)
            if disease_context and count > 0:
                evidence_score *= 1.2  # Bonus for disease context
            
            # Fetch article details for top results
            articles = self.fetch_article_details(ids[:5]) if ids else []
            
            result = {
                'feature_name': feature_name,
                'paper_count': count,
                'sample_ids': ids[:5],
                'articles': articles,  # Now includes full article details
                'evidence_score': round(evidence_score, 1),
                'search_query': search_term,
                'disease_context': disease_context
            }
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logging.error(f"PubMed search error for {feature_name}: {e}")
            st.warning(f"PubMed search error for {feature_name}: {str(e)}")
            return {
                'feature_name': feature_name,
                'paper_count': 0,
                'sample_ids': [],
                'articles': [],
                'evidence_score': 0.0,
                'search_query': '',
                'disease_context': disease_context
            }
    
    def batch_search(self, features: List[str], disease_context: str = None, 
                    progress_callback=None) -> List[dict]:
        """Batch search for multiple features"""
        results = []
        total = len(features)
        
        for i, feature in enumerate(features):
            if progress_callback:
                progress_callback(i + 1, total, feature)
            
            result = self.search_simple(feature, disease_context)
            results.append(result)
            
        return results