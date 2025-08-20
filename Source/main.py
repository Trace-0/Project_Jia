import threading
from discord_interface import discordBot
from discord_interface.pipeline import app
from discord_interface.discordBot import app as disapp
import uvicorn

def run_pipeline():
    uvicorn.run(app=app, host="127.0.0.1", port=8000)

def run_discordBot():
    uvicorn.run(app=disapp, host="127.0.0.1", port=8001)
threading.Thread(target=run_pipeline, daemon=True).start()
threading.Thread(target=run_discordBot, daemon=True).start()

discordBot.bot()