import logging
import choko_model
import time
import os

from aiogram import Bot, Dispatcher, executor, types

#API_TOKEN = '1011117684:AAG4jD1d7p6N19DNV7NqXr3kvhN0bdZVbN8'
API_TOKEN = os.eviron['BOT_TOKEN']
USE_WEBHOOK = os.eviron['USE_WEBHOOK']

# webhook settings
WEBHOOK_HOST = 'https://ala-foto-bot.herokuapp.com/'
WEBHOOK_PATH = os.eviron['BOT_TOKEN']
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

# webserver settings
WEBAPP_HOST = '0.0.0.0'  # or ip
WEBAPP_PORT = int(os.eviron['PORT'])

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
save_path = 'data/'
model = choko_model.Choko_transform_model(save_path)

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """
    This handler will be called when user sends `/start` or `/help` command
    """
    if USE_WEBHOOK == '1':
        message_hook = ' (using webhook)'
    else:
        message_hook = ''
        
    await message.reply("Hi!\nI'm AlaFotoBot' + message_hook + '! I can make your foto look different!\nSend me a foto and see yourself!")

@dp.message_handler(content_types=['photo'])
async def foto_input(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)
    # message.answer('Nice foto! Wait a sec, we are transforming it.')
    ms = int(time.time())

    file_name = 'foto' + str(ms) + '.jpg'
    await message.photo[-1].download(save_path + file_name)
    out_path = model.Transform_to_choco(file_name)
    #message.answer('Here!')
    #await message.answer_photo(save_path + 'conv_'+ file_name)
    img_result = open(out_path, 'rb')
    await message.answer_photo(img_result)
    


@dp.message_handler()
async def echo(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)

    await message.answer(message.text)

    
async def on_startup(dp):
    await bot.set_webhook(WEBHOOK_URL)
    # insert code here to run it after start


async def on_shutdown(dp):
    logging.warning('Shutting down..')

    # insert code here to run it before shutdown

    # Remove webhook (not acceptable in some cases)
    await bot.delete_webhook()

    # Close DB connection (if used)
    await dp.storage.close()
    await dp.storage.wait_closed()

    logging.warning('Bye!')


if __name__ == '__main__':
    if USE_WEBHOOK == '1':
        executor.start_webhook(
            dispatcher=dp,
            webhook_path=WEBHOOK_PATH,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            skip_updates=True,
            host=WEBAPP_HOST,
            port=WEBAPP_PORT,
        )
     else:
        executor.start_polling(dp, skip_updates=True)
        
