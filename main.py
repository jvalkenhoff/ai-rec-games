import os
import requests
from pathlib import Path
from getpass import getpass
from dotenv import load_dotenv
from playwright.sync_api import (
  sync_playwright,
  Page,
  Locator,
  TimeoutError as PlaywrightTimeoutError,
  )

from playwright.sync_api import Locator
from openai import OpenAI
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL = "gpt-4.1-mini"
MAX_GAMES = 75

PARAMS = {
  'specials': '1',
  'supportedlang': 'english',
  'hwtype': '0',
  'category1': '998',
  'category3': '1',
  'ndl': '1'
}

SYSTEM_PROMPT = """
You are an experienced video game critic specializing in evaluating, comparing, and recommending video games across different genres, platforms, and budgets.

Your job is to curate a personalized list of video game recommendations based on the user's preferences, requirements, and budget. Prioritize games that offer the best overall experience and value for money.

Focus Areas:
- Genre: Gameplay style, mechanics, and genre compatibility.
- Price & Value: Price relative to content, quality, and replayability.
- Release Date: Match the user's preferred release period.
- Quality: Critical reception, player reviews, and overall polish.
- Content: Story, gameplay depth, progression, and estimated playtime.
- Preferences: Match the user's specific interests, platforms, and restrictions.

Rules:
1. Recommend 5-10 games ranked by how well they match the user's criteria.
2. Respect hard constraints such as budget, platform, and release year.
3. Prioritize quality and value over popularity or commercial success.
4. Provide a short explanation of why each game is recommended.
5. Include release year, genre, estimated playtime, and price when available.
6. Never invent prices, ratings, or game details. Clearly distinguish verified information from estimates.
7. If current pricing is unavailable, explicitly say so rather than assuming a discount.
8. Present recommendations in a clear Markdown table, followed by a brief Top Pick explaining the best overall choice.
9. Avoid unnecessary introductions and keep recommendations concise and objective.
"""

def build_url() -> str:
  request = requests.Request('GET', 'https://store.steampowered.com/search/', params=PARAMS)
  prepared_request = request.prepare()
  return prepared_request.url

def check_apikey() -> str:
  load_dotenv(override=True)
  api_key = os.getenv("OPENAI_API_KEY") or getpass("OpenAI API Key (not stored): ")

  if not api_key:
    raise RuntimeError("An OpenAI API key is required")
  elif not api_key.startswith("sk-proj-"):
    raise RuntimeError("API Key must start with 'sk-proj-'")
  elif api_key.strip() != api_key:
    raise RuntimeError("API key contains leading or trailing whitespace")

  print("API Key looks good!")
  return api_key

def openai_client(api_key: str) -> OpenAI:
  return OpenAI(api_key=api_key)

def get_text(row: Locator, selector: str) -> str:
  element = row.locator(selector)

  if element.count() == 0:
    return ""

  return element.first.inner_text().strip()

def parse_reviews(row: Locator) -> str | None:
  rating = row.locator(".search_review_summary")

  if rating.count() == 0:
    return None

  tooltip = rating.first.get_attribute("data-tooltip-html")
  return tooltip.split("<br", 1)[0].strip() if tooltip else None

def parse_game(row: Locator) -> dict:
  return {
    "App ID": row.get_attribute("data-ds-appid"),
    "Title": get_text(row, ".title"),
    "Release Date": get_text(row, ".search_released"),
    "Discount": get_text(row, ".discount_pct"),
    "Price": get_text(row, ".discount_final_price"),
    "URL": row.get_attribute("href"),
    "Reviews": parse_reviews(row),
  }

def load_page(page: Page, url: str) -> Locator:
  """Loads the full discount page first. Locator makes sure the important DOM elements are actually present."""
  response = page.goto(url, wait_until='domcontentloaded')
  if not response or not response.ok:
    raise RuntimeError("Failed to load Steam website")

  rows = page.locator("a.search_result_row")
  rows.first.wait_for(state='visible', timeout=20000)
  print("Games are visible and ready to scrape!")
  return rows

def collect_games(rows: Locator, start: int, games: list[dict], max_games: int) -> int:
  """Parse rows from `start` onward into `games`. Returns the new processed count."""
  count = rows.count()

  for i in range(start, count):
    if len(games) >= max_games:
      break

    game = parse_game(rows.nth(i))
    if game["Title"] and game["URL"]:
      games.append(game)

  return count

def load_more_games(page: Page, rows: Locator) -> bool:
  """Scroll behavior to trigger infinite loading if necessary. Returns True if news rows appeared."""
  previous_count = rows.count()
  rows.last.scroll_into_view_if_needed()
  page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

  try:
    print("We have to scroll, please wait...")
    page.wait_for_function(
      f"previous => document.querySelectorAll('a.search_result_row').length > previous",
      arg=previous_count,
      timeout=8000,
    )
    return True
  except PlaywrightTimeoutError:
    print("No additional games loaded")
    return False

def scrape_games(url: str, max_games: int) -> list[dict]:
  games: list[dict] = []
  processed_count = 0

  with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)

    try:
      page = browser.new_context(locale="en-US").new_page()
      rows = load_page(page=page, url=url)

      while len(games) < max_games:
        processed_count = collect_games(rows, processed_count, games, max_games)
        print(f"Collected {len(games)} games")

        if len(games) >= max_games or not load_more_games(page, rows):
          break
    finally:
      browser.close()
  return games


def recommend_games(df: pd.DataFrame, client: OpenAI) -> str:
  user_prompt = """
  Recommend up to 5 games that offer the best value.

  Preferences:
  - Maximum price: 40 EUR
  - Prefer games released from 2015 onward
  - Favor strong reviews and substantial discounts

  Use only the provided dataset.
  Do not assume genres, playtime, or ratings
  that are not included in the data.
  Include the Steam URL for each recommendation.

  Games:
  """

  json_df = df.to_json(orient='records', force_ascii=False)

  response = client.responses.create(
    model=MODEL,
    instructions=SYSTEM_PROMPT,
    input=user_prompt + json_df,
    store=False
  )

  return response.output_text

def main() -> None:
  try:
    api_key = check_apikey()
    client = openai_client(api_key)

    print("OpenAI client initialized!")
  except RuntimeError as e:
    print(f"Configuration error: {e}")
    return
  except Exception as e:
    print(f"Unexpected error: {e}")
    raise

  url = build_url()
  print(f"Scraping: {url}")
  games = scrape_games(url, MAX_GAMES)

  if not games:
    raise RuntimeError("No games collected")

  df = pd.DataFrame(games)
  print("AI Recommendations:\n")
  print(recommend_games(df=df, client=client))


if __name__ == "__main__":
  main()