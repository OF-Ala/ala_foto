import logging
import choko_model
import time

from aiogram import Bot, Dispatcher, executor, types

API_TOKEN = '1011117684:AAG4jD1d7p6N19DNV7NqXr3kvhN0bdZVbN8'

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
    await message.reply("Hi!\nI'm AlaFotoBot! I can make your foto look more choko!\nSend me a foto and see yourself!")

@dp.message_handler(content_types=['photo'])
async def foto_input(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)
    await message.answer('Nice foto! Wait a sec, we are transforming it.')
    ms = int(time.time())

    file_name = 'foto' + ms + '.jpg'
    message.photo[-1].download(save_path + file_name)
    model.Transform_to_choko(file_name)
    await message.answer_photo(save_path + 'conv_'+ file_name)
    await message.answer('Here!')


@dp.message_handler()
async def echo(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)

    await message.answer(message.text)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
