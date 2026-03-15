import httpx

from app.core.config import get_settings
from app.services.disease_catalog import DISEASE_CATALOG, get_disease_info


class GrokService:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def get_disease_description(self, disease_name: str) -> tuple[str, bool]:
        fallback = DISEASE_CATALOG.get(
            disease_name,
        )
        if fallback:
            fallback_text = fallback.get("description", "This result should be reviewed by a qualified medical professional for proper diagnosis.")
        else:
            fallback_text = "This result should be reviewed by a qualified medical professional for proper diagnosis."
        
        if not self.settings.grok_api_key:
            return fallback_text, True

        payload = {
            "model": self.settings.grok_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You explain predicted skin diseases in simple language. "
                        "Respond in 2 to 3 sentences. Avoid claiming certainty. "
                        "Mention that clinical confirmation is important."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Explain the skin disease prediction: {disease_name}.",
                },
            ],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self.settings.grok_api_key}"}

        try:
            async with httpx.AsyncClient(base_url=self.settings.grok_base_url, timeout=20.0) as client:
                response = await client.post("/chat/completions", json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                return (content or fallback_text), (not bool(content))
        except Exception:
            return fallback_text, True

    async def get_detailed_report(self, disease_name: str) -> dict:
        """Get detailed disease report with symptoms and recommendations using Grok"""
        # Get base disease info from catalog
        disease_info = get_disease_info(disease_name)
        
        if not self.settings.grok_api_key:
            # Return catalog info with fallback message
            return {
                "disease_name": disease_info.get("name", disease_name.replace("_", " ")),
                "description": disease_info.get("description", ""),
                "symptoms": disease_info.get("symptoms", []),
                "risk_factors": disease_info.get("risk_factors", []),
                "recommendations": disease_info.get("recommendations", []),
                "when_to_see_doctor": disease_info.get("when_to_see_doctor", "Consult a healthcare professional"),
                "generated_by": "Catalog"
            }

        # Use Grok to enhance the report
        payload = {
            "model": self.settings.grok_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical information assistant specializing in skin diseases. "
                        "Provide detailed, accurate information about the skin condition. "
                        "Always include a disclaimer that this is not medical advice. "
                        "Format your response as a JSON object with: "
                        "- description: Brief overview (2-3 sentences) "
                        "- symptoms: Array of common symptoms "
                        "- risk_factors: Array of risk factors "
                        "- recommendations: Array of self-care recommendations "
                        "- when_to_see_doctor: When to seek professional help "
                        "Use the disease name provided."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Provide detailed information about {disease_name} skin condition in JSON format.",
                },
            ],
            "temperature": 0.3,
        }
        headers = {"Authorization": f"Bearer {self.settings.grok_api_key}"}

        try:
            async with httpx.AsyncClient(base_url=self.settings.grok_base_url, timeout=30.0) as client:
                response = await client.post("/chat/completions", json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                
                # Try to parse JSON from response
                import json
                import re
                
                # Find JSON in response (in case there's text around it)
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    grok_data = json.loads(json_match.group())
                    
                    # Merge with catalog data (catalog takes precedence for structure)
                    return {
                        "disease_name": disease_info.get("name", disease_name.replace("_", " ")),
                        "description": grok_data.get("description", disease_info.get("description", "")),
                        "symptoms": grok_data.get("symptoms", disease_info.get("symptoms", [])),
                        "risk_factors": grok_data.get("risk_factors", disease_info.get("risk_factors", [])),
                        "recommendations": grok_data.get("recommendations", disease_info.get("recommendations", [])),
                        "when_to_see_doctor": grok_data.get("when_to_see_doctor", disease_info.get("when_to_see_doctor", "Consult a healthcare professional")),
                        "generated_by": "Grok AI"
                    }
        except Exception as e:
            print(f"Error getting detailed report from Grok: {e}")
        
        # Fallback to catalog info
        return {
            "disease_name": disease_info.get("name", disease_name.replace("_", " ")),
            "description": disease_info.get("description", ""),
            "symptoms": disease_info.get("symptoms", []),
            "risk_factors": disease_info.get("risk_factors", []),
            "recommendations": disease_info.get("recommendations", []),
            "when_to_see_doctor": disease_info.get("when_to_see_doctor", "Consult a healthcare professional"),
            "generated_by": "Catalog (Grok unavailable)"
        }


grok_service = GrokService()
