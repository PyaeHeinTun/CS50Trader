from beta_bot.bot_start import bot_start
import asyncio
from beta_bot.helper import base as helper
from beta_bot.temp_storage import TempStorage, temp_storage_data

async def main() -> None:
    config = helper.read_config()
    temp_storage_data[TempStorage.config] = config

    bot_task = asyncio.create_task(bot_start())

    done, pending = await asyncio.wait([bot_task], return_when=asyncio.FIRST_COMPLETED)

    for task in pending:
        task.cancel()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass