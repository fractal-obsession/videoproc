import json
import requests
import random
import argparse
import os
import logging
from moviepy.editor import *
import PIL
import numpy
import tempfile


parser = argparse.ArgumentParser(description='ComfyUI tools')
parser.add_argument('command', type=str, default='run',
                    help='workflow json in api format')
# general
parser.add_argument('-c', '--checkpoint', type=str, default='bluePencilXL_v200.safetensors', help='dir for images')
parser.add_argument('-w', '--prompt_workflow', type=str, default='i2i_api.json', help='workflow json in api format (not implemented)')
parser.add_argument('-s', '--steps', type=int, default=30, help='steps total')
parser.add_argument('-D', '--denoise', type=float, default=0.5, help='start step')
parser.add_argument('-p', '--prompt', type=str, default='a cow in a dungeon', help='positive prompt')
parser.add_argument('-n', '--noise', type=int, default=random.randint(1, 18446744073709551614), help='noise seed')
parser.add_argument('-C', '--cfg', type=float, default=10, help='cfg')
parser.add_argument('-v', '--verbose', action='store_true', help='verbosity')
# i2i
parser.add_argument('-d', '--dir', type=str, default='~/ai/ComfyUI/input/batch', help='dir for images')

# vid
parser.add_argument('-V', '--video', type=str, default='~/ai/ComfyUI/input/vid1/video1.mp4', help='video')
parser.add_argument('--start_time', type=float, default=None, help='start time')
parser.add_argument('--end_time', type=float, default=None, help='end time')
parser.add_argument('--start_frame', type=int, default=0, help='start_frame')
parser.add_argument('--frame_step', type=int, default=1, help='frame step')
parser.add_argument('--end_frame', type=int, default=-1, help='frame end')
parser.add_argument('-t', '--tempdir', default='/tmp/videoproc', help='tempdir')

# canny
parser.add_argument('--strength', type=float, default=1, help='controllnet strength')

args = parser.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)

def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    req =  requests.post("http://127.0.0.1:8188/prompt", data=data)
    logging.info("{}".format(req.json()))

def vid2frames():
    #tempdir = tempfile.TemporaryDirectory(delete=False)
    if not os.path.exists(args.tempdir):
            os.makedirs(args.tempdir)

    if args.end_time and args.start_time:
        logging.info("subclip {} to {}".format(args.start_time, args.end_time))
        penis = args.end_time
        vid = VideoFileClip(args.video, audio_buffersize=500000).subclip(args.start_time, args.end_time)
    else:
        vid = VideoFileClip(args.video, audio_buffersize=500000)

    logging.info("clip duration is {}".format(vid.duration))
    audio = vid.audio.to_soundarray(fps=vid.fps, buffersize=500000)
    logging.info("audio is {}".format(audio))
    images = []

    for index,frame in enumerate(vid.iter_frames(with_times=True)):
        logging.info('index: {}'.format(index))
        if args.end_frame > 0 and index >= args.end_frame:
            logging.info(images)
            return images
        if index >= args.start_frame:
            frame_save_path = os.path.join(args.tempdir, "frame_{}.png".format(index))
            frame_volume = 0
            frame_image = PIL.Image.fromarray(frame[1])
            frame_image.save(frame_save_path)
            images.append((index, frame_save_path, frame_volume))
            logging.info('Read a new frame: {}'.format(frame_save_path))
    logging.info(images)
    return images

def command_status():
    req =  requests.get("http://127.0.0.1:8188/queue")
    print("{} running".format(req.json()['queue_running'].__len__()))
    for job in req.json()['queue_running']:
        logging.info(job[1])
    print("{} pending".format(req.json()['queue_pending'].__len__()))
    for job in req.json()['queue_pending']:
        logging.info(job[1])

def command_runv_canny():
    prompt_workflow = json.load(open('i2i_canny_api.json'))
    prompt_workflow['63']['inputs']['steps'] = args.steps
    prompt_workflow['63']['inputs']['denoise'] = args.denoise
    prompt_workflow['4']['inputs']['ckpt_name'] = args.checkpoint
    prompt_workflow['6']['inputs']['text'] = args.prompt
    prompt_workflow['7']['inputs']['text'] = "watermark"
    prompt_workflow['64']['inputs']['strength'] = args.strength
    if args.noise == -1:
        prompt_workflow['63']['inputs']['seed'] = random.randint(1, 18446744073709551614)
    else:
        prompt_workflow['63']['inputs']['seed'] = args.noise
    images = vid2frames()
    for i in range(0, len(images), args.frame_step):
        prompt_workflow['19']['inputs']['filename_prefix'] = "{}_frame{}".format(args.command, images[i][0])
        prompt_workflow['50']['inputs']['image'] = images[i][1]
        prompt_workflow['69']['inputs']['image'] = images[i][1]
        queue_prompt(prompt_workflow)
        print("{}".format(images[i][1]))

def command_runv():
    prompt_workflow = json.load(open('i2i_api.json'))
    prompt_workflow['10']['inputs']['steps'] = args.steps
    prompt_workflow['10']['inputs']['denoise'] = args.denoise
    prompt_workflow['4']['inputs']['ckpt_name'] = args.checkpoint
    prompt_workflow['6']['inputs']['text'] = args.prompt
    prompt_workflow['7']['inputs']['text'] = "watermark"
    if args.noise == -1:
        prompt_workflow['10']['inputs']['seed'] = random.randint(1, 18446744073709551614)
    else:
        prompt_workflow['10']['inputs']['_seed'] = args.noise
    images = vid2frames()
    for i in range(0, len(images), args.frame_step):
        prompt_workflow['19']['inputs']['filename_prefix'] = "{}_frame{}".format(args.command, images[i][0])
        prompt_workflow['50']['inputs']['image'] = images[i][1]
        queue_prompt(prompt_workflow)
        print("{}".format(images[i][1]))

def command_run():
    prompt_workflow = json.load(open('i2i_api.json'))
    prompt_workflow['10']['inputs']['steps'] = args.steps
    prompt_workflow['10']['inputs']['denoise'] = args.denoise
    prompt_workflow['4']['inputs']['ckpt_name'] = args.checkpoint
    prompt_workflow['6']['inputs']['text'] = args.prompt
    prompt_workflow['7']['inputs']['text'] = "watermark"
    if args.noise == -1:
        prompt_workflow['10']['inputs']['seed'] = random.randint(1, 18446744073709551614)
    else:
        prompt_workflow['10']['inputs']['_seed'] = args.noise
    if os.path.isfile(args.dir):
        prompt_workflow['50']['inputs']['image'] = os.path.join(args.dir)
        queue_prompt(prompt_workflow)
    else:
        for root, dirs, files in os.walk(args.dir):
            for name in files:
                print("{}/{}".format(root, name))
                prompt_workflow['50']['inputs']['image'] = os.path.join(root, name)
                queue_prompt(prompt_workflow)

if args.command == "run":
    command_run()
elif args.command == "status":
    command_status()
elif args.command == "runv":
    command_runv()
elif args.command == "runv_canny":
    command_runv_canny()
