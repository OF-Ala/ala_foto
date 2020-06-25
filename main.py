import logging
import transform_model
import time
import os
import hair_classifier

from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton

API_TOKEN = os.environ['BOT_TOKEN']
USE_WEBHOOK = os.environ['USE_WEBHOOK']

# webhook settings
WEBHOOK_HOST = 'https://ala-foto-bot.herokuapp.com'
WEBHOOK_PATH = f"/{API_TOKEN}/"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

# webserver settings
WEBAPP_HOST = '0.0.0.0'  # or ip
WEBAPP_PORT = int(os.environ['PORT'])

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
save_path = 'data/'
model_blond2brown = transform_model.Transform_model(save_path, 'blonde2brown')
model_blond2ginger = transform_model.Transform_model(save_path, 'blonde2ginger')
model_brown2ginger = transform_model.Transform_model(save_path, 'brown2ginger')

model_black2brown = transform_model.Transform_model(save_path, 'black2brown')
model_black2ginger = transform_model.Transform_model(save_path, 'black2ginger')
model_black2blond = transform_model.Transform_model(save_path, 'black2blonde')

hair_classifier = hair_classifier.Classifier_model(save_path)

colors_dict = {
  0: 'black',  
  1: 'blonde',
  2: 'brown',
  3: 'ginger' 
}

files_color_dict = {}  #input_color

inline_btn_black = InlineKeyboardButton('Black', callback_data=0)
inline_btn_blonde = InlineKeyboardButton('Blonde', callback_data=1)
inline_btn_brown = InlineKeyboardButton('Brown', callback_data=2)
inline_btn_ginger = InlineKeyboardButton('Ginger', callback_data=3)

inline_kb_from0 = InlineKeyboardMarkup().add(inline_btn_brown, inline_btn_blonde,inline_btn_ginger)
inline_kb_from1 = InlineKeyboardMarkup().add(inline_btn_brown, inline_btn_ginger,inline_btn_black)
inline_kb_from2 = InlineKeyboardMarkup().add(inline_btn_blonde, inline_btn_ginger,inline_btn_black)
inline_kb_from3 = InlineKeyboardMarkup().add(inline_btn_brown, inline_btn_blonde,inline_btn_black)

transform_func_dict = {
    (0,2):model_black2brown.Transform_to_B, #black 2 brown
    (0,3):model_black2ginger.Transform_to_B, #black 2 ginger
    (0,1):model_black2blond.Transform_to_B, #black to blond
    
    (1,2):model_blond2brown.Transform_to_B, #blond 2 brown
    (1,3):model_blond2ginger.Transform_to_B, #blond 2 ginger
    (1,0):model_black2blond.Transform_to_A, #blond to black
    
    (2,1):model_blond2brown.Transform_to_A, #brown 2 blond
    (2,3):model_brown2ginger.Transform_to_B, #brown to ginger
    (2,0):model_black2brown.Transform_to_A, #brown 2 black
    
    (3,2):model_brown2ginger.Transform_to_A, #ginger 2 brown
    (3,1):model_blond2ginger.Transform_to_A, #ginger 2 blond
    (3,0):model_black2ginger.Transform_to_A #ginger 2 black
}

kb_dict = {
    0:inline_kb_from0,
    1:inline_kb_from1,
    2:inline_kb_from2,
    3:inline_kb_from3    
}
    

def Get_input_color(file_name):
    input_color, confidence = hair_classifier.predict_one_sample(file_name)
    files_color_dict[file_name] = input_color
    return input_color, confidence


def Make_transformation(file_name, input_color, output_color):
    if (input_color, output_color) in transform_func_dict:
        output_name = transform_func_dict[(input_color, output_color)](file_name)
    else:
        output_name = self.save_path + file_name

    return output_name

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """
    This handler will be called when user sends `/start` or `/help` command
    """
    if USE_WEBHOOK == '1':
        message_hook = ' (using webhook)'
    else:
        message_hook = ''

    await message.reply(
        "Hi!\nI'm AlaFotoBot' + message_hook + '! I can make your foto look different!\nSend me a foto and see yourself!")


@dp.message_handler(content_types=['photo'])
async def foto_input(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)
    # message.answer('Nice foto! Wait a sec, we are transforming it.')
    ms = int(time.time())

    file_name = 'foto' + str(message.chat.id) + '.jpg'
    await message.photo[-1].download(save_path + file_name)
    foto_color, confidence = Get_input_color(file_name)
    foto_color_name = colors_dict[foto_color]
    if confidence > 0.85:
        answer_str = f"Nice foto! Your hair color is {foto_color_name}!\nChoose color you want to switch to:"
    elif confidence > 0.65:
        answer_str = f"Nice foto! Looks like your hair color is {foto_color_name}.\nChoose color you want to switch to:"
    else:
        answer_str = f"Nice foto! Your hair color looks like {foto_color_name}. I\'m not sure though.\nChoose color you want to switch to:"

    if foto_color in kb_dict:
        await message.answer(answer_str, reply_markup=kb_dict[foto_color])
    else:
        await message.answer("Something went wrong, sorry. Answer:", foto_color)


#@dp.callback_query_handler(func=lambda c: c.data)
@dp.callback_query_handler(lambda callback_query: True)
async def process_callback_transform(callback_query: types.CallbackQuery):
    output_color = int(callback_query.data)
    file_name = 'foto' + str(callback_query.from_user.id) + '.jpg'
    input_color = files_color_dict[file_name]

    out_path = Make_transformation(file_name, input_color, output_color)

    img_result = open(out_path, 'rb')
    await bot.send_photo(callback_query.from_user.id, img_result, \
                         "Changing hair color from " + colors_dict[input_color] + " to " + colors_dict[output_color])
    img_result = None


@dp.message_handler()
async def echo(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)

    await message.answer("Hi!\nI'm AlaFotoBot! I can make your foto look different!\nSend me a foto and see yourself!")


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


def webhook(update):
    dispatcher.process_update(update)


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
