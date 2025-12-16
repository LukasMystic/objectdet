from datetime import datetime, timedelta

async def cleanup_unverified_users(db):
    """
    Removes users who are unverified and created > 24 hours ago.
    This function is isolated to avoid circular imports in main.py.
    """
    cutoff = datetime.utcnow() - timedelta(hours=24)
    result = await db.users.delete_many({
        "is_verified": False,
        "created_at": {"$lt": cutoff}
    })
    if result.deleted_count > 0:
        print(f"Startup Cleanup: Removed {result.deleted_count} unverified users.")