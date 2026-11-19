"""Pygame viewer for manual, scripted, and trained controllers."""
import argparse
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import pygame
from environment import EscapeRoom, IDS, scripted, random_room
import secrets

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--controller',choices=['manual','scripted','ppo','options','free','map'],default='scripted')
    parser.add_argument('--checkpoint'); parser.add_argument('--seed',type=int)
    parser.add_argument('--screenshot'); parser.add_argument('--frames',type=int,default=0)
    args=parser.parse_args()
    round_seed = args.seed if args.seed is not None else secrets.randbits(32)
    policies=None; communication=True
    if args.controller=='map':
        from map_ppo import load_map, room_for, moves
        map_policy = load_map()
    if args.controller=='free':
        import json
        from pathlib import Path
        from free_roles import Policy, setup
        path=Path(args.checkpoint or 'runs/free_roles/seed7.json')
        free_policy=Policy(json.loads(path.read_text())['genome'])
    if args.controller=='options':
        import json
        from pathlib import Path
        from options_rl import Controller
        if not args.checkpoint: parser.error('--checkpoint is required for learned goals')
        data=json.loads(Path(args.checkpoint).read_text())
        communication=data['channel']
        learned_goals=Controller(data['tables'],communication)
    if args.controller=='ppo':
        if not args.checkpoint: parser.error('--checkpoint is required for PPO')
        from learning import load,choose
        import torch
        torch.manual_seed(round_seed)
        policies,data=load(args.checkpoint); communication=data['config']['communication']
    def start_round(seed):
        if args.controller=='map':
            room = room_for(seed)
            return room, room.reset(seed)[0]
        if args.controller=='free':
            return setup(seed)
        config = random_room(seed) if args.controller != 'ppo' else None
        room = EscapeRoom(communication, config)
        return room, room.reset(seed)[0]
    env, obs = start_round(round_seed)
    pygame.init(); screen=pygame.display.set_mode((1080,660)); pygame.display.set_caption('Cooperative Escape Room | AI Research MVP')
    font=pygame.font.SysFont('segoeui',19); small=pygame.font.SysFont('segoeui',15); title=pygame.font.SysFont('segoeui',28,bold=True)
    clock=pygame.time.Clock(); selected=0; pending=[0,0]; paused=False; speed=4; accumulator=0; running=True; frames=0
    colors=[(66,182,255),(255,173,79)]
    def text(s,x,y,color=(221,231,242),f=None): screen.blit((f or font).render(s,True,color),(x,y))
    def advance(actions=None):
        nonlocal obs
        if env.done:return
        if actions is None:
            walls=env.config.get('walls', ())
            if args.controller=='map':
                actions=moves(map_policy,obs)
            elif args.controller=='free':
                actions=free_policy.act(obs)
            elif args.controller=='options':
                actions=learned_goals.act(obs,walls)
            elif args.controller=='scripted':
                actions=scripted(obs,walls)
            else:
                actions=choose(policies,obs)
        obs,_,_,_,_=env.step(actions)
    while running:
        dt=clock.tick(60)/1000; accumulator+=dt
        for event in pygame.event.get():
            if event.type==pygame.QUIT: running=False
            if event.type==pygame.KEYDOWN:
                key=event.key
                if getattr(event,'unicode','').lower() in ['r','p','n','e','0','1','2']:
                    key=ord(event.unicode.lower())
                if key==pygame.K_ESCAPE: running=False
                elif key==pygame.K_r:
                    round_seed += 1
                    env,obs=start_round(round_seed); accumulator=0; pending=[0,0]
                elif key==pygame.K_TAB:selected=1-selected
                elif key==pygame.K_p: paused=not paused
                elif key in [pygame.K_EQUALS,pygame.K_PLUS,pygame.K_KP_PLUS]:speed=min(32,speed*2)
                elif key in [pygame.K_MINUS,pygame.K_KP_MINUS]:speed=max(1,speed//2)
                elif key==pygame.K_n and args.controller!='manual':paused=True; advance()
                elif key in [pygame.K_0,pygame.K_1,pygame.K_2]:pending[selected]=key-pygame.K_0
                elif args.controller=='manual' and not paused:
                    mapping={pygame.K_UP:0,pygame.K_DOWN:1,pygame.K_LEFT:2,pygame.K_RIGHT:3,pygame.K_SPACE:4,pygame.K_e:5}
                    if key in mapping:
                        actions={k:(4,pending[i]) for i,k in enumerate(IDS)}
                        actions[IDS[selected]]=(mapping[key],pending[selected]); advance(actions)
        if args.controller!='manual' and not paused:
            if env.done and accumulator>=2:
                round_seed += 1
                env,obs=start_round(round_seed); accumulator=0
            elif not env.done and accumulator>=1/speed:
                advance(); accumulator=0
        screen.fill((13,20,31)); text('COOPERATIVE / ESCAPE ROOM',28,22,f=title)
        text(f'Room seed {round_seed}',28,65,color=(139,159,182))
        tile=64; ox=28; oy=116
        for y in range(7):
            for x in range(9):
                rect=pygame.Rect(ox+x*tile,oy+y*tile,tile-3,tile-3)
                wall=x in (0,8) or y in (0,6) or [x,y] in env.config.get('walls', ())
                pygame.draw.rect(screen,(42,55,73) if wall else (24,35,49),rect,border_radius=5)
        objects=[(env.config['plate'],'PLATE',(121,94,183)),(env.config['switches'][0],'1',(193,150,57)),(env.config['switches'][1],'2',(193,150,57)),(env.config['exit'],'EXIT',(53,167,121) if env.unlocked else (149,62,75))]
        for (x,y),label,color in objects:
            r=pygame.Rect(ox+x*tile+5,oy+y*tile+5,tile-13,tile-13)
            pygame.draw.rect(screen,color,r,border_radius=7)
            text(label,r.x+5,r.y+3,f=small)
        for i,(x,y) in enumerate(env.pos):
            if env.escaped[i]:continue
            center=(ox+x*tile+30,oy+y*tile+37)
            pygame.draw.circle(screen,colors[i],center,17)
            if args.controller=='manual' and selected==i:pygame.draw.circle(screen,(245,247,250),center,22,2)
            text(IDS[i],center[0]-6,center[1]-13,color=(13,20,31))
        panel=pygame.Rect(638,116,414,448); pygame.draw.rect(screen,(21,31,45),panel,border_radius=12)
        label={'manual':'Manual','scripted':'Scripted','ppo':'Trained PPO','options':'Learned cooperation','free':'Learned moves','map':'Full-map PPO'}[args.controller]
        text(label.upper(),660,136,color=(86,213,185),f=title)
        reader=IDS[getattr(env,'clue_holder',0)]
        lines=[f'Step {env.steps:03d} / 100   |   {speed} steps/sec',f'Status: {env.outcome.upper()}',f'Exit: {"UNLOCKED" if env.unlocked else "LOCKED"}',f'Agent {reader} private clue: switch {env.clue+1}',f'Messages sent: A={env.messages[0]}  B={env.messages[1]}',f'Channel: {"enabled" if communication else "disabled"}',f'Escaped: A={env.escaped[0]}  B={env.escaped[1]}']
        for j,line in enumerate(lines): text(line,660,185+j*32)
        rule='Either agent can hold the plate or use a switch.' if args.controller in ('map','free') else 'A holds the plate. B activates the right switch.'
        text(rule,660,425,f=small)
        text('Then BOTH agents reach the exit.',660,449,f=small)
        text('Observer view: policies do not see this panel.',660,491,color=(139,159,182),f=small)
        finished = 'R for next room' if args.controller=='manual' else 'Next round starts automatically'
        text('PAUSED' if paused else (finished if env.done else 'RUNNING'),660,524,color=(255,206,109),f=small)
        text('R next round  •  P pause  •  N single step  •  + / − playback speed  •  Esc quit',28,584,f=small)
        if args.controller=='manual':
            text('Tab select agent  •  Arrows move  •  E interact  •  Space wait  •  0/1/2 message',28,612,f=small)
        pygame.display.flip(); frames+=1
        if args.screenshot and frames==1: pygame.image.save(screen,args.screenshot)
        if args.frames and frames>=args.frames:running=False
    pygame.quit()

if __name__=='__main__':main()
