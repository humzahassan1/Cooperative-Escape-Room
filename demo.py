"""Pygame viewer for manual, scripted, and trained controllers."""
import argparse
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import pygame
from environment import EscapeRoom, IDS, scripted, random_room
import secrets

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--controller',choices=['manual','scripted','ppo','options','free','map','roles','plate'],default='scripted')
    parser.add_argument('--checkpoint'); parser.add_argument('--seed',type=int)
    parser.add_argument('--replay')
    parser.add_argument('--campaign',action='store_true',help='Play the vault followed by the relay chamber')
    parser.add_argument('--relay',action='store_true',help='Start directly in the relay chamber')
    parser.add_argument('--screenshot'); parser.add_argument('--frames',type=int,default=0)
    args=parser.parse_args()
    if (args.campaign or args.relay) and args.controller not in ('scripted','manual'):
        parser.error('Relay uses seven actions; use manual or scripted playback.')
    from relay_room import RelayRoom
    round_seed = args.seed if args.seed is not None else secrets.randbits(32)
    policies=None; communication=True
    role_queue = []
    role_hidden = None
    role_generator = None
    role_inference = None
    if args.controller=='roles':
        import json
        from pathlib import Path
        import torch
        from role_ppo import load_role, case_generator, reset_for_evaluation
        from environment import random_room as role_room
        checkpoint = args.checkpoint or 'runs/role_coordination/seed7_comm/checkpoint.pt'
        if not Path(checkpoint).exists():
            parser.error(f'role checkpoint not found: {checkpoint}')
        role_model, role_payload = load_role(checkpoint)
        communication = role_payload['communication']
        role_hidden = torch.zeros(2, role_model.gru.hidden_size)
        role_generator = torch.Generator().manual_seed(round_seed)
        interpretation_path = Path(checkpoint).with_name('interpretation.json')
        if not interpretation_path.exists():
            interpretation_path = Path('runs/role_coordination/interpretation.json')
        role_inference = json.loads(interpretation_path.read_text()) if interpretation_path.exists() else None
        if args.replay:
            replay = json.loads(Path(args.replay).read_text())
            role_queue = list(replay.get('episodes', []))
    if args.controller=='map':
        from map_ppo import load_map, room_for, moves
        map_policy = load_map()
    if args.controller=='free':
        import json
        from pathlib import Path
        from free_roles import Policy, setup
        path=Path(args.checkpoint or 'runs/free_roles/seed7.json')
        free_policy=Policy(json.loads(path.read_text())['genome'])
    if args.controller=='plate':
        from plate_signal import PlateRoom, plate_actions, load_tables
        plate_tables = load_tables()
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
        if args.controller=='plate':
            room = PlateRoom()
            return room, room.reset(seed)[0]
        nonlocal role_hidden, role_generator
        if args.relay:
            room=RelayRoom(seed)
            return room,room.observations()
        if args.controller=='roles':
            import torch
            from role_ppo import HIDDEN, case_generator, reset_for_evaluation
            if role_queue:
                episode = role_queue.pop(0)
                room = reset_for_evaluation(episode['config'], episode['clue'], episode['holder'], episode['swap'], communication, episode['layout_seed'])
                room.replay_actions = episode.get('actions')
            else:
                room = reset_for_evaluation(role_room(seed), None, None, None, communication, seed)
                room.replay_actions = None
            room.inference = role_inference
            room.replay_index = 0
            role_hidden = torch.zeros(2, HIDDEN)
            role_generator = case_generator(seed, getattr(room, 'clue', 0), getattr(room, 'clue_holder', 0), 0)
            return room, room.observations()
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
        nonlocal obs, role_hidden
        if env.done:return
        if actions is None:
            walls=env.config.get('walls', ())
            if isinstance(env,RelayRoom):
                actions=env.reference_actions()
            elif args.controller=='roles':
                recorded = getattr(env, 'replay_actions', None)
                if recorded is not None and env.replay_index < len(recorded):
                    actions = {key: tuple(value) for key, value in recorded[env.replay_index].items()}
                    env.replay_index += 1
                else:
                    from role_ppo import policy_forward, sample_actions
                    move_logits, message_logits, _, updated = policy_forward(role_model, [obs], role_hidden[None])
                    actions = sample_actions(move_logits[0], message_logits[0], role_generator)
                    role_hidden = updated[0].detach()
            elif args.controller=='plate':
                actions = plate_actions(plate_tables, obs, walls)
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
        mode = {'manual':'Manual','scripted':'Scripted','ppo':'Trained PPO','options':'Learned cooperation','free':'Learned moves','map':'Full-map PPO','plate':'Plate signal demo','roles':'Role coordination' if communication else 'Role coordination silent'}[args.controller]
        view.draw(screen, env, mode, round_seed, speed, paused, selected)
        pygame.display.flip(); frames+=1
        if args.screenshot and frames==1: pygame.image.save(screen,args.screenshot)
        if args.frames and frames>=args.frames:running=False
    pygame.quit()

if __name__=='__main__':main()
