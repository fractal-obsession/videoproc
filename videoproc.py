#!/usr/bin/python3
import json
import requests
import random
import argparse
import os
import sys
import logging
from moviepy.editor import *
import PIL
import numpy
import tempfile
import numpy as np
import time
import webvtt


parser = argparse.ArgumentParser(description='ComfyUI tools', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('command', type=str, default='run',
                    help='which mode to use: status(display queue status), clist(list available checkpoints), run(image2image), runv(video2images), run_prompt_blend')
# general
parser.add_argument('-r', '--rootdir', type=str, default=os.path.join(os.environ['HOME'], "ai/ComfyUI/"), help='ComfyUI install directory')
parser.add_argument('-c', '--checkpoint', type=str, default=None, help='checkpoint name to use, needs to be in ComfyUI/models/checkpoints/')
parser.add_argument('-w', '--prompt_workflow', type=str, default='i2i_api.json', help='workflow json in api format (not implemented)')
parser.add_argument('-s', '--steps', type=int, default=30, help='total denoising steps, this does not change with denoising levels')
parser.add_argument('-d', '--denoise', type=float, default=0.5, help='denoising degree, 0 returns input image, 1 completely ignores input image')
parser.add_argument('-P', '--prompt', type=str, default='a cow in a dungeon', help='positive prompt')
parser.add_argument('-N', '--negative_prompt', type=str, default='disfigured, deformed, ugly', help='negative prompt')
parser.add_argument('-D', '--dynamic_prompt', type=str, default='a cow in a dungeon', help='dynamic positive prompt')
parser.add_argument('-m', '--prompt_mix', type=float, default=0, help='promt mixing coeficient, 0 uses exclusively \'positive prompt\', 1 uses exclusively \'dynamic prompt\'')
parser.add_argument('-n', '--noise', type=int, default=random.randint(1, 18446744073709551614), help='noise seed, random if ommited')
parser.add_argument('-C', '--cfg', type=float, default=10, help='CFG Scale, also known as Configuration Scale, is a parameter in Stable Diffusion that affects how accurately the AI-generated image aligns with the original text prompt.')
parser.add_argument('-v', '--verbose', action='store_true', help='verbosity')
parser.add_argument('--dry_run', action='store_true', help='do not submit to api or create images form video frames')
parser.add_argument('-o', '--outdir', type=str, default='', help='dir for output images')

# i2i
parser.add_argument('-p', '--path', type=str, default=os.path.join(os.environ['HOME'], 'ai/ComfyUI/input/batch'), help='input image, or directory with images')

# prompt_blend
parser.add_argument('--blend_steps', type=int, default=10, help='total number of steps for the prompt blending')

# vid
parser.add_argument('-V', '--video', type=str, default=os.path.join(os.environ['HOME'], 'ai/ComfyUI/input/vid/video1.mp4'), help='path to video file')
parser.add_argument('-S', '--subs', type=str, default=None, help='path to vtt subtiles file')
parser.add_argument('--start_time', type=float, default=None, help='time in seconds(float) from which to start in the video')
parser.add_argument('--end_time', type=float, default=None, help='time in seconds(float) up to (excluding) which to process video')
parser.add_argument('--start_frame', type=int, default=0, help='frame from the subclip returned by --start/end_time to start from')
parser.add_argument('--frame_step', type=int, default=1, help='step between frames, usefull for initial experimentation')
parser.add_argument('--end_frame', type=int, default=-1, help='frame from the subclip returned by --start/end_time to end on')
parser.add_argument('-t', '--tempdir', default='/tmp/videoproc', help='directory to store extracted frames before procession by the workflow')
parser.add_argument('--audio_modulate', action='store_true', help='audio volume modulation')

# canny
parser.add_argument('--canny_strength', type=float, default=0.6, help='strength of the applied canny controllnet from 0 to 1')
parser.add_argument('--canny_low', type=float, default=0.05, help='canny filter low threshold')
parser.add_argument('--canny_high', type=float, default=0.2, help='canny filter high threshold')

args = parser.parse_args()
if args.checkpoint:
    if not args.checkpoint in os.listdir(os.path.join(args.rootdir, "models/checkpoints/")):
        logging.error("checkpoint must be one of:")
        for checkpoint in os.listdir(os.path.join(args.rootdir, "models/checkpoints/")):
            logging.error(checkpoint)
        sys.exit(1)
else:
    args.checkpoint = os.listdir(os.path.join(args.rootdir, "models/checkpoints/"))[0]


if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)

def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    if not args.dry_run:
        req =  requests.post("http://127.0.0.1:8188/prompt", data=data)
        logging.info("{}".format(req.json()))

def vid2frames():
    #tempdir = tempfile.TemporaryDirectory(delete=False)
    if not args.dry_run:
        if not os.path.exists(args.tempdir):
                os.makedirs(args.tempdir)

    if args.end_time and args.start_time:
        logging.info("subclip {} to {}".format(args.start_time, args.end_time))
        vid = VideoFileClip(args.video, audio_buffersize=5000000).subclip(args.start_time, args.end_time)
    else:
        vid = VideoFileClip(args.video, audio_buffersize=5000000)

    if args.subs:
        subs = webvtt.read(args.subs).captions
        logging.info("read subs file {}".format(args.subs))
    else:
        subs = []
    for sub_line in subs:
        print("{} - {}: {}".format(sub_line.start_in_seconds, sub_line.end_in_seconds, sub_line.text))


    logging.info("clip duration is {}".format(vid.duration))
    audio = [item[0] for item in abs(vid.audio.to_soundarray(fps=vid.fps, buffersize=5000000, quantize=False))]
    audio = np.convolve(audio, np.ones(15), 'same') / 15
    audio = audio/max(audio)
    logging.info("got {} audio frames and {} video frames with max(audio) {} and min(audio) {}".format(len(audio), vid.duration * vid.fps, max(audio), min(audio)))
    frames = []

    for index,frame in enumerate(vid.iter_frames(with_times=True)):
        frame_obj = {}
        frame_obj['index'] = index
        frame_obj['time'] = frame[0] + args.start_time
        frame_obj['volume'] = audio[frame_obj['index']]
        frame_obj['path'] = os.path.join(args.tempdir, "frame_{}.png".format(frame_obj['index']))

        for sub_line in subs:
            if sub_line.start_in_seconds <=frame_obj['time'] < sub_line.end_in_seconds:
                frame_obj['subtitle'] = sub_line.text
                break
            else:
                frame_obj['subtitle'] = ""

        logging.info('frame: {}'.format(frame_obj))

        if args.end_frame > 0 and frame_obj['index'] >= args.end_frame:
            logging.info(frames)
            return frames
        if frame_obj['index'] >= args.start_frame and not (frame_obj['index'] % args.frame_step):
            if not args.dry_run:
                PIL.Image.fromarray(frame[1]).save(frame_obj['path'])
            frames.append(frame_obj)
            logging.info('Read(past tense) a new frame: {}'.format(frame_obj['path']))
    logging.info(frames)
    return frames

def config_prompt_workflow(prompt_workflow):
    prompt_workflow['63']['inputs']['steps'] = args.steps
    prompt_workflow['63']['inputs']['denoise'] = args.denoise
    prompt_workflow['63']['inputs']['cfg'] = args.cfg
    prompt_workflow['4']['inputs']['ckpt_name'] = args.checkpoint
    prompt_workflow['6']['inputs']['text'] = args.prompt
    prompt_workflow['7']['inputs']['text'] = args.negative_prompt
    prompt_workflow['75']['inputs']['text'] = args.dynamic_prompt
    prompt_workflow['74']['inputs']['conditioning_to_strength'] = args.prompt_mix
    prompt_workflow['64']['inputs']['strength'] = args.canny_strength
    prompt_workflow['73']['inputs']['low_threshold'] = args.canny_low
    prompt_workflow['73']['inputs']['high_threshold'] = args.canny_high
    if args.noise == -1:
        prompt_workflow['63']['inputs']['seed'] = random.randint(1, 18446744073709551614)
    else:
        prompt_workflow['63']['inputs']['seed'] = args.noise
    return prompt_workflow

def command_status():
    req =  requests.get("http://127.0.0.1:8188/queue")
    running = req.json()['queue_running'].__len__()
    pending = req.json()['queue_pending'].__len__()
    total = req.json()['queue_running'].__len__() + req.json()['queue_pending'].__len__()
    print("{} running {} pending {} total".format(running, pending, total))
    for job in req.json()['queue_running']:
        logging.info("running")
        logging.info(job[1])
    for job in req.json()['queue_pending']:
        logging.info("pending")
        logging.info(job[1])

def command_checkpoint_list():
    for checkpoint in os.listdir(os.path.join(args.rootdir, "models/checkpoints/")):
        print(checkpoint)

def command_runv():
    logging.info("running command_runv_canny")
    prompt_workflow = json.load(open('i2i_canny_api.json'))
    prompt_workflow = config_prompt_workflow(prompt_workflow)
    frames = vid2frames()
    for frame in frames:
        prompt_workflow['19']['inputs']['filename_prefix'] = os.path.join(args.outdir, "{}_{:05d}_{}".format(int(time.time()), frame['index'], args.command))
        prompt_workflow['50']['inputs']['image'] = frame['path']
        prompt_workflow['69']['inputs']['image'] = frame['path']
        if 'subtitle' in frame.keys():
            prompt_workflow['75']['inputs']['text'] = frame['subtitle']
        if args.audio_modulate:
            prompt_workflow['63']['inputs']['denoise'] = args.denoise * frame['volume']
            logging.info("denoising after audio modulation {}".format(args.denoise * frame['volume']))
        queue_prompt(prompt_workflow)
        print("{}".format(frame['index']))

def command_run():
    logging.info("running command_run_canny")
    prompt_workflow = json.load(open('i2i_canny_api.json'))
    prompt_workflow = config_prompt_workflow(prompt_workflow)
    if os.path.isfile(args.path):
        prompt_workflow['50']['inputs']['image'] = os.path.join(args.path)
        prompt_workflow['69']['inputs']['image'] = os.path.join(args.path)
        queue_prompt(prompt_workflow)
    else:
        print(args.path)
        for root, dirs, files in os.walk(args.path):
            for name in files:
                print("{}/{}".format(root, name))
                prompt_workflow['19']['inputs']['filename_prefix'] = os.path.join(args.outdir, "{}_{}".format(int(time.time()), args.command))
                prompt_workflow['50']['inputs']['image'] = os.path.join(root, name)
                prompt_workflow['69']['inputs']['image'] = os.path.join(root, name)
                queue_prompt(prompt_workflow)

def command_run_prompt_blend():
    logging.info("running command_run_canny")
    prompt_workflow = json.load(open('i2i_canny_api.json'))
    prompt_workflow = config_prompt_workflow(prompt_workflow)
    if os.path.isfile(args.path):
        for i in [x/float(args.blend_steps) for x in range(0, args.blend_steps + 1)]:
            prompt_workflow['74']['inputs']['conditioning_to_strength'] = i
            logging.info("mixing with factor {}".format(i))
            prompt_workflow['50']['inputs']['image'] = os.path.join(args.path)
            prompt_workflow['69']['inputs']['image'] = os.path.join(args.path)
            queue_prompt(prompt_workflow)
    else:
        logging.warning("please provide a path to one image file, yo")

if args.command == "run":
    command_run()
elif args.command == "status":
    command_status()
elif args.command == "runv":
    command_runv()
elif args.command == "runv_canny":
    command_runv_canny()
elif args.command == "run_prompt_blend":
    command_run_prompt_blend()
elif args.command == "clist":
    command_checkpoint_list()
