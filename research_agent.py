import nest_asyncio
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.config import Config
from newspaper import Article
import re

nest_asyncio.apply()

class ResearchAgent:
    def __init__(self):
        self.search_tool = TavilySearchResults(
            api_key=Config.TAVILY_API_KEY,
            max_results=Config.MAX_SEARCH_RESULTS
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=200
        )

    async def gather_information(self, query: str):
        search_results = await self.search_tool.ainvoke(query)
        urls = [r['url'] for r in search_results]
        loader = AsyncHtmlLoader(urls)
        docs = await loader.aload()

        readable_docs = []
        for idx, url in enumerate(urls):
            try:
                article = Article(url)
                article.download()
                article.parse()
                clean_text = article.text
                if len(clean_text.strip()) < 300:
                    continue
                doc = docs[idx]
                doc.page_content = clean_text
                readable_docs.append(doc)
            except Exception as e:
                print(f"[Error] {url}: {e}")

        chunks = self.text_splitter.split_documents(readable_docs)
        return self._filter_noise(chunks)

    def _filter_noise(self, chunks):
        noise_patterns = [
            r'\[.*?\]', r'\{.*?\}', r'<.*?>',
            r'cookie|consent|preferences|marketing',
            r'read more|view book|follow me',
            r'padding|margin|border|background|color|font',
            r'fusion_builder|admin_label|type="flex"',
            r'[A-Za-z_]+="[^"]*"',
        ]
        patterns = [re.compile(p, re.IGNORECASE) for p in noise_patterns]

        junk_phrases = [
            "sign up", "newsletter", "email", "privacy policy",
            "terms & conditions", "captcha", "submit", "view preferences"
        ]

        cleaned = []
        for doc in chunks:
            text = doc.page_content.lower()
            if len(text) < 300 or any(phrase in text for phrase in junk_phrases):
                continue
            for pat in patterns:
                text = pat.sub('', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) < 300:
                continue
            doc.page_content = text
            cleaned.append(doc)
        return cleaned