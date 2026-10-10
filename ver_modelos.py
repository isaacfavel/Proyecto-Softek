from groq import Groq

# Pon tu API key real aquí
client = Groq(api_key="")

try:
    modelos = client.models.list()
    print("Modelos activos en tu cuenta:")
    for m in modelos.data:
        print(f"- {m.id}")
except Exception as e:
    print(f"Error: {e}")