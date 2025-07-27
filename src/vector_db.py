import qdrant_client

import config

client = qdrant_client.QdrantClient(config.QDRANT_URL)
