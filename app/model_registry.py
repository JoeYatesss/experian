import os
import json
from typing import Dict, Optional, List
from datetime import datetime

class ModelRegistry:
    def __init__(self, registry_dir: str = "models"):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
        
    def get_latest_model(self) -> Optional[Dict]:
        """Get the latest model version info"""
        versions = self.list_versions()
        if not versions:
            # If no models in registry, try to use the root model file
            if os.path.exists("xgboost.json"):
                return {
                    'version': 'default',
                    'model_path': "xgboost.json",
                    'metadata': {'source': 'root_directory'}
                }
            return None
            
        latest_version = versions[-1]  # Versions are sorted by timestamp
        return self.get_model_info(latest_version)
    
    def list_versions(self) -> List[str]:
        """List all available model versions"""
        if not os.path.exists(self.registry_dir):
            return []
            
        versions = []
        for version_dir in os.listdir(self.registry_dir):
            if os.path.isdir(os.path.join(self.registry_dir, version_dir)):
                versions.append(version_dir)
                
        # Sort versions by timestamp (assuming YYYYMMDD_HHMMSS format)
        return sorted(versions)
    
    def get_model_info(self, version: str) -> Optional[Dict]:
        """Get model info for a specific version"""
        version_dir = os.path.join(self.registry_dir, version)
        metadata_path = os.path.join(version_dir, 'metadata.json')
        model_path = os.path.join(version_dir, 'model.json')
        
        if not os.path.exists(metadata_path) or not os.path.exists(model_path):
            return None
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        return {
            'version': version,
            'model_path': model_path,
            'metadata': metadata
        }
    
    def register_model(self, model_path: str, metadata: Dict) -> str:
        """Register a new model version"""
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_dir = os.path.join(self.registry_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # Copy model file
        new_model_path = os.path.join(version_dir, 'model.json')
        with open(model_path, 'rb') as src, open(new_model_path, 'wb') as dst:
            dst.write(src.read())
        
        # Save metadata
        metadata_path = os.path.join(version_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return version 