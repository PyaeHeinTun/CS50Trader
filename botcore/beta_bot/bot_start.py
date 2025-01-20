from beta_bot.controller import base as controller


async def bot_start() -> None:
    await controller.fetch_multiple_coins()
