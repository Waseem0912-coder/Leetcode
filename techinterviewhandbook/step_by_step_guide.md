You can handle duplicates and multiple content entries for the same ID using several approaches. Here are the most effective strategies:

## 1. Composite Primary Key + Unique Constraint

Modify your table structure to handle multiple content entries per ID:

```python
# models.py
from django.db import models
import hashlib

class ScrapedData(models.Model):
    ticket_id = models.CharField(max_length=100)  # ID from email
    content = models.TextField()
    content_hash = models.CharField(max_length=64, unique=True)  # SHA-256 hash
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['ticket_id', 'content_hash']  # Prevent exact duplicates
        indexes = [
            models.Index(fields=['ticket_id']),
            models.Index(fields=['content_hash']),
        ]

    def save(self, *args, **kwargs):
        # Generate hash of content for deduplication
        self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        super().save(*args, **kwargs)
```

## 2. Smart Insertion Logic

Implement deduplication logic in your scraper:

```python
# scraper.py
import hashlib
from django.db import IntegrityError
from .models import ScrapedData

def save_scraped_data(ticket_id, content):
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    
    try:
        # Try to create new entry
        obj, created = ScrapedData.objects.get_or_create(
            ticket_id=ticket_id,
            content_hash=content_hash,
            defaults={'content': content}
        )
        
        if created:
            print(f"New content saved for ticket {ticket_id}")
        else:
            print(f"Duplicate content ignored for ticket {ticket_id}")
            
        return obj, created
        
    except IntegrityError:
        # Handle race conditions
        print(f"Integrity error for ticket {ticket_id}")
        return None, False
```

## 3. Content Versioning Approach

If you want to track content changes over time:

```python
class ScrapedData(models.Model):
    ticket_id = models.CharField(max_length=100)
    content = models.TextField()
    content_hash = models.CharField(max_length=64)
    version = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    is_latest = models.BooleanField(default=True)
    
    class Meta:
        unique_together = ['ticket_id', 'content_hash']
        indexes = [
            models.Index(fields=['ticket_id', 'is_latest']),
        ]

def save_with_versioning(ticket_id, content):
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    
    # Check if this exact content already exists
    if ScrapedData.objects.filter(ticket_id=ticket_id, content_hash=content_hash).exists():
        return None, False
    
    # Mark previous entries as not latest
    ScrapedData.objects.filter(ticket_id=ticket_id, is_latest=True).update(is_latest=False)
    
    # Get next version number
    last_version = ScrapedData.objects.filter(ticket_id=ticket_id).aggregate(
        max_version=models.Max('version')
    )['max_version'] or 0
    
    # Create new entry
    obj = ScrapedData.objects.create(
        ticket_id=ticket_id,
        content=content,
        content_hash=content_hash,
        version=last_version + 1,
        is_latest=True
    )
    
    return obj, True
```

## 4. Batch Processing with Bulk Operations

For better performance when processing many emails:

```python
from django.db import transaction

def process_emails_batch(email_data_list):
    new_entries = []
    existing_hashes = set()
    
    # Get existing content hashes for all ticket IDs in batch
    ticket_ids = [item['ticket_id'] for item in email_data_list]
    existing_hashes = set(
        ScrapedData.objects.filter(ticket_id__in=ticket_ids)
        .values_list('content_hash', flat=True)
    )
    
    for item in email_data_list:
        content_hash = hashlib.sha256(item['content'].encode()).hexdigest()
        
        if content_hash not in existing_hashes:
            new_entries.append(ScrapedData(
                ticket_id=item['ticket_id'],
                content=item['content'],
                content_hash=content_hash
            ))
            existing_hashes.add(content_hash)  # Prevent duplicates within batch
    
    # Bulk create new entries
    if new_entries:
        with transaction.atomic():
            ScrapedData.objects.bulk_create(new_entries, ignore_conflicts=True)
    
    return len(new_entries)
```

## 5. Query Helpers

Add useful methods to retrieve data:

```python
class ScrapedDataManager(models.Manager):
    def get_latest_for_ticket(self, ticket_id):
        return self.filter(ticket_id=ticket_id, is_latest=True).first()
    
    def get_all_versions_for_ticket(self, ticket_id):
        return self.filter(ticket_id=ticket_id).order_by('-version')

class ScrapedData(models.Model):
    # ... your fields ...
    objects = ScrapedDataManager()
```

## Recommended Approach

I'd suggest using **approach #1 (Composite Primary Key)** with **approach #2 (Smart Insertion Logic)** because:

- It prevents exact duplicates efficiently
- Allows multiple different content entries per ticket ID
- Uses database constraints for data integrity
- Simple to implement and maintain
- Good performance with proper indexing

The content hash ensures you never store identical content twice, while still allowing legitimate updates to the same ticket ID.
