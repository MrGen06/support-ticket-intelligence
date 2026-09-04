from datetime import datetime, timedelta, timezone

SLA_HOURS = {
    "Critical": 4,
    "High": 8,
    "Medium": 24,
    "Low": 72
}

def calculate_sla(priority, received_at=None):
    # Using timezone-aware UTC as requested by the PDF's production note
    received_at = received_at or datetime.now(timezone.utc)
    
    # Fallback to Medium if priority string doesn't perfectly match
    hours = SLA_HOURS.get(priority, 24) 
    deadline = received_at + timedelta(hours=hours)
    
    return {
        "priority": priority,
        "sla_hours": hours,
        "deadline": deadline.isoformat()
    }