from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

class Database:
    client: AsyncIOMotorClient = None
    db = None

    def connect(self):
        """Connect to MongoDB using the env vars"""
        if not settings.MONGODB_URL:
            print("MongoDB URL not set in .env")
            return

        print(f"Connecting to MongoDB at {settings.MONGODB_URL.split('@')[-1]}") # Log safe part of URL
        self.client = AsyncIOMotorClient(settings.MONGODB_URL)
        self.db = self.client[settings.DB_NAME]
        print(f"Connected to database: {settings.DB_NAME}")

    def close(self):
        if self.client:
            self.client.close()
            print("Disconnected from MongoDB")

db = Database()

async def get_db():
    return db.db