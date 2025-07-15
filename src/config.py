import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler("sansad_qa.log"), logging.StreamHandler()],
)
logger = logging.getLogger("sansad_qa")


class Config:
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Directory structure
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    PDF_CACHE_DIR = DATA_DIR / "pdf_cache"
    MINISTRY_PDF_DIR = DATA_DIR / "ministry_pdfs"
    VECTOR_DB_DIR = DATA_DIR / "vector_db"

    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # API settings
    SANSAD_API_URL = "https://sansad.in/api_ls/question/qetFilteredQuestionsAns"
    PDF_BASE_URL = "https://sansad.in/"
    DEFAULT_LOK_SABHA = 18
    DEFAULT_SESSION = 4
    DEFAULT_PAGE_SIZE = 100

    # Processing settings
    MAX_RETRIES = 3
    TIMEOUT = 30
    RATE_LIMIT_DELAY = 1
    MAX_DOCS_PER_QUERY = 10
    PDF_BATCH_SIZE = 20

    # Complete list of ministries
    MINISTRIES = [
        "Ministry of Agriculture and Farmers Welfare",
        "Ministry of Chemicals and Fertilizers",
        "Ministry of Civil Aviation",
        "Ministry of Coal",
        "Ministry of Commerce and Industry",
        "Ministry of Communications",
        "Ministry of Consumer Affairs, Food and Public Distribution",
        "Ministry of Corporate Affairs",
        "Ministry of Culture",
        "Ministry of Defence",
        "Ministry of Development of North Eastern Region",
        "Ministry of Earth Sciences",
        "Ministry of Education",
        "Ministry of Electronics and Information Technology",
        "Ministry of Environment, Forest and Climate Change",
        "Ministry of External Affairs",
        "Ministry of Finance",
        "Ministry of Fisheries, Animal Husbandry and Dairying",
        "Ministry of Food Processing Industries",
        "Ministry of Health and Family Welfare",
        "Ministry of Heavy Industries",
        "Ministry of Home Affairs",
        "Ministry of Housing and Urban Affairs",
        "Ministry of Information and Broadcasting",
        "Ministry of Jal Shakti",
        "Ministry of Labour and Employment",
        "Ministry of Law and Justice",
        "Ministry of Micro, Small and Medium Enterprises",
        "Ministry of Mines",
        "Ministry of Minority Affairs",
        "Ministry of New and Renewable Energy",
        "Ministry of Panchayati Raj",
        "Ministry of Parliamentary Affairs",
        "Ministry of Personnel, Public Grievances and Pensions",
        "Ministry of Petroleum and Natural Gas",
        "Ministry of Power",
        "Ministry of Railways",
        "Ministry of Road Transport and Highways",
        "Ministry of Rural Development",
        "Ministry of Science and Technology",
        "Ministry of Ports, Shipping and Waterways",
        "Ministry of Skill Development and Entrepreneurship",
        "Ministry of Social Justice and Empowerment",
        "Ministry of Statistics and Programme Implementation",
        "Ministry of Steel",
        "Ministry of Textiles",
        "Ministry of Tourism",
        "Ministry of Tribal Affairs",
        "Ministry of Women and Child Development",
        "Ministry of Youth Affairs and Sports",
        "Prime Minister's Office",
        "NITI Aayog",
    ]

    @staticmethod
    def sanitize_ministry_name(ministry):
        """Convert ministry name to directory-safe format"""
        return ministry.replace(" ", "_").replace(",", "").replace("'", "")

    @classmethod
    def get_ministry_dir(cls, ministry):
        """Get directory path for a specific ministry"""
        dir_name = cls.sanitize_ministry_name(ministry)
        return cls.MINISTRY_PDF_DIR / dir_name

    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist"""
        # Create main directories
        cls.PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.MINISTRY_PDF_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

        # Create directory for each ministry
        for ministry in cls.MINISTRIES:
            ministry_dir = cls.get_ministry_dir(ministry)
            ministry_dir.mkdir(exist_ok=True)

        # Create unknown ministry directory
        unknown_dir = cls.MINISTRY_PDF_DIR / "Unknown_Ministry"
        unknown_dir.mkdir(exist_ok=True)

        logger.info("All required directories created")

    @classmethod
    def validate_environment(cls):
        """Validate environment configuration"""
        if not cls.GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not found in environment variables")
            return False

        return True


def _save_indexed_ministries(self):
    """Save information about indexed ministries to a metadata file"""
    try:
        metadata_path = Path(Config.VECTOR_DB_DIR) / "indexed_ministries.json"

        # Generate timestamp directly
        from datetime import datetime
        current_time = datetime.now().isoformat()
        
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "ministries": list(self.indexed_ministries),
                    "updated_at": current_time,
                    "updated_by": os.getenv("USERNAME", "anonymous"),
                },
                f,
                indent=2,
            )

        logger.info(f"Saved {len(self.indexed_ministries)} indexed ministries to metadata")

    except Exception as e:
        logger.warning(f"Error saving indexed ministries: {e}")