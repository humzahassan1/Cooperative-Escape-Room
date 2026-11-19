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
    parser.add_argument('--campaign',action='store_true',help='Play the vault followed by the relay chamber')
    parser.add_argument('--relay',action='store_true',help='Start directly in the relay chamber')
    parser.add_argument('--screenshot'); parser.add_argument('--frames',type=int,default=0)
    args=parser.parse_args()
    if (args.campaign or args.relay) and args.controller not in ('scripted','manual'):
        parser.error('Relay uses seven actions; use manual or scripted playback.')
    from relay_room import RelayRoom
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
        if args.relay:
            room=RelayRoom(seed)
            return room,room.observations()
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
    from vault_view import VaultView
    view = VaultView()
    clock=pygame.time.Clock(); selected=0; pending=[0,0]; paused=False; speed=4; accumulator=0; running=True; frames=0
    def advance(actions=None):
        nonlocal obs
        if env.done:return
        if actions is None:
            walls=env.config.get('walls', ())
            if isinstance(env,RelayRoom):
                actions=env.reference_actions()
            elif args.controller=='map':
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
    def next_room():
        nonlocal env,obs,round_seed,accumulator,pending
        if args.campaign and not isinstance(env,RelayRoom) and env.outcome=='success':
            env=RelayRoom(round_seed)
            obs=env.observations()
        else:
            round_seed+=1
            env,obs=start_round(round_seed)
        accumulator=0
        pending=[0,0]
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
                    next_room()
                elif key==pygame.K_TAB:selected=1-selected
                elif key==pygame.K_p: paused=not paused
                elif key in [pygame.K_EQUALS,pygame.K_PLUS,pygame.K_KP_PLUS]:speed=min(32,speed*2)
                elif key in [pygame.K_MINUS,pygame.K_KP_MINUS]:speed=max(1,speed//2)
                elif key==pygame.K_n and args.controller!='manual':paused=True; advance()
                elif key in [pygame.K_0,pygame.K_1,pygame.K_2]:pending[selected]=key-pygame.K_0
                elif args.controller=='manual' and not paused:
                    mapping={pygame.K_UP:0,pygame.K_DOWN:1,pygame.K_LEFT:2,pygame.K_RIGHT:3,pygame.K_SPACE:4,pygame.K_e:5}
                    if isinstance(env,RelayRoom): mapping[pygame.K_f]=6
                    if key in mapping:
                        actions={k:(4,pending[i]) for i,k in enumerate(IDS)}
                        actions[IDS[selected]]=(mapping[key],pending[selected]); advance(actions)
        if args.controller!='manual' and not paused:
            if env.done and accumulator>=2:
                next_room()
            elif not env.done and accumulator>=1/speed:
                advance(); accumulator=0
        mode = {'manual':'Manual','scripted':'Scripted','ppo':'Trained PPO','options':'Learned cooperation','free':'Learned moves','map':'Full-map PPO'}[args.controller]
        view.draw(screen, env, mode, round_seed, speed, paused, selected)
        pygame.display.flip(); frames+=1
        if args.screenshot and frames==1: pygame.image.save(screen,args.screenshot)
        if args.frames and frames>=args.frames:running=False
    pygame.quit()

if __name__=='__main__':main()
