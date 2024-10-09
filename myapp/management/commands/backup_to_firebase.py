import json
from django.core.management.base import BaseCommand
from firebase_admin import firestore
from myapp.models import models  # Replace with your actual model

class Command(BaseCommand):
    help = 'Backup SQLite data to Firebase Firestore'

    def handle(self, *args, **kwargs):
        # Initialize Firestore
        db = firestore.client()

        # Fetch data from SQLite database
        data = models.objects.all()

        # Backup data to Firebase Firestore
        for item in data:
            item_dict = item.__dict__
            item_dict.pop('_state', None)  # Remove internal state

            # Convert to JSON serializable format if needed
            item_dict = json.loads(json.dumps(item_dict))

            # Add to Firestore
            db.collection('MyModelCollection').document(str(item.pk)).set(item_dict)

        self.stdout.write(self.style.SUCCESS('Successfully backed up data to Firebase'))
