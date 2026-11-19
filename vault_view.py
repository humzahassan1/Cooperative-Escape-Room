"""Pygame art and observer instruments for the submerged observatory."""
import math
from pathlib import Path
import pygame


class VaultView:
    def __init__(self):
        folder = Path(__file__).parent / 'assets' / 'vault'
        self.sprites = {}
        for name,size in dict(diver=52,crab=59,wall=64,floor=64,plate=49,
                              switch=45,shut=57,open=57).items():
            source = pygame.image.load(folder / f'{name}.png').convert_alpha()
            w,h = source.get_size()
            dims = (64,64) if name in ('floor','wall') else (round(size*w/max(w,h)),round(size*h/max(w,h)))
            self.sprites[name] = pygame.transform.smoothscale(source,dims)
        self.font = pygame.font.SysFont('consolas',15,bold=True)
        self.name = pygame.font.SysFont('georgia',38,bold=True)
        self.medium = pygame.font.SysFont('segoeui',22,bold=True)
        self.colors = [(103,218,210),(255,173,84)]
        self.board_key = None
        self.room = None

    def draw(self, screen, env, mode, seed, speed, paused, selected):
        ink=(225,227,207); muted=(153,178,177); brass=(211,169,99)
        def write(s,x,y,color=ink,font=None):
            screen.blit((font or self.font).render(str(s),True,color),(x,y))
        def sprite(name,x,y):
            img=self.sprites[name]
            screen.blit(img,img.get_rect(center=(round(x),round(y))))
        screen.fill((10,23,29))
        pygame.draw.rect(screen,(14,34,40),(0,0,1080,96))
        pygame.draw.line(screen,(87,104,87),(0,95),(1080,95))
        write('COOPERATIVE ESCAPE ROOM',26,11,brass)
        relay=hasattr(env,'grille')
        write('The Relay Chamber' if relay else 'The Sunken Vault',24,29,ink,self.name)
        write(f'ROOM  {seed}',650,24,muted)
        write(mode.upper(),650,47,self.colors[0],self.medium)
        ox,oy=26,117
        pygame.draw.rect(screen,(123,106,75),(ox-4,oy-4,584,456),border_radius=4)
        key=tuple(map(tuple,env.config.get('walls',())))
        if key!=self.board_key:
            self.board_key=key
            self.board=pygame.Surface((576,448))
            self.board.fill((11,28,34))
            for y in range(7):
                for x in range(9):
                    wall=x in (0,8) or y in (0,6) or (x,y) in key
                    self.board.blit(self.sprites['wall' if wall else 'floor'],(64*x,64*y))
        screen.blit(self.board,(ox,oy))
        cfg=env.config
        def center(p):
            return ox+p[0]*64+32,oy+p[1]*64+32
        if relay:
            x,y=center(env.socket)
            sprite('plate',x,y)
            write('SOCKET',x-25,y+16,ink)
            gx,gy=center(env.grille)
            if not env.unlocked:
                pygame.draw.rect(screen,(13,27,33),(gx-31,gy-12,62,24))
                for offset in range(-24,30,10):
                    pygame.draw.line(screen,brass,(gx+offset,gy-11),(gx+offset,gy+11),3)
            if env.carrier is None and not env.unlocked:
                fx,fy=center(env.fuse)
                pygame.draw.rect(screen,(160,237,211),(fx-14,fy-7,28,14),border_radius=3)
                for offset in (-18,12):
                    pygame.draw.rect(screen,brass,(fx+offset,fy-10,6,20))
        else:
            px,py=center(cfg['plate'])
            active=tuple(cfg['plate']) in env.pos
            if active: pygame.draw.circle(screen,(170,132,221),(px,py),28,3)
            sprite('plate',px,py)
        for number,p in enumerate(cfg.get('switches',()),1):
            x,y=center(p); sprite('switch',x,y)
            pygame.draw.rect(screen,(20,29,33),(x+13,y-28,17,18),border_radius=3)
            write(number,x+17,y-28,brass)
        sprite('open' if env.unlocked else 'shut',*center(cfg['exit']))
        now=pygame.time.get_ticks()
        if self.room is not env or env.steps==0:
            self.room=env
            self.previous=list(env.pos); self.current=list(env.pos); self.moved=now
        elif self.current!=env.pos:
            self.previous=self.current; self.current=list(env.pos); self.moved=now
        blend=min(1,(now-self.moved)/min(140,850/speed))
        for i,kind in enumerate(('diver','crab')):
            if env.escaped[i]: continue
            a,b=self.previous[i],self.current[i]
            x,y=center((a[0]+(b[0]-a[0])*blend,a[1]+(b[1]-a[1])*blend))
            x,y=round(x),round(y)
            pygame.draw.ellipse(screen,(8,20,25),(x-20,y+15,40,12))
            if mode=='Manual' and selected==i:
                pygame.draw.ellipse(screen,self.colors[i],(x-26,y+12,52,17),2)
            sprite(kind,x,y+round(1.5*math.sin(now/350+i)))
            if relay and env.carrier==i:
                pygame.draw.rect(screen,(160,237,211),(x+13,y-22,18,8),border_radius=2)
        pygame.draw.rect(screen,(18,39,45),(628,115,428,450),border_radius=8)
        write('CREW TELEMETRY',648,134,brass)
        for i,(kind,caption) in enumerate([('diver','A   Brass Diver'),('crab','B   Crab Scout')]):
            top=165+68*i
            sprite(kind,679,top+24)
            write(caption,722,top,self.colors[i],self.medium)
            write('ESCAPED' if env.escaped[i] else f'Transmitted symbol   {env.messages[i]}',723,top+31,muted)
        pygame.draw.line(screen,(58,85,87),(648,308),(1034,308))
        reader='AB'[getattr(env,'clue_holder',0)]
        write(env.event if relay else f'Private clue     {reader} sees switch {env.clue+1}',650,329)
        write(f'Channel          {"enabled" if env.communication else "disabled"}',650,357)
        if relay:
            fuse_state='installed' if env.unlocked else ('at supply' if env.carrier is None else 'with ' + 'AB'[env.carrier])
            write(f'Relay fuse       {fuse_state}',650,385)
        else:
            write(f'Pressure plate   {"occupied" if active else "empty"}',650,385)
        write(f'Exit hatch       {"OPEN" if env.unlocked else "sealed"}',650,413)
        write(f'Step {env.steps:03d} / {cfg["max_steps"]}     {speed} steps/sec',650,452,muted)
        pygame.draw.rect(screen,(39,63,68),(650,478,382,5))
        pygame.draw.rect(screen,brass,(650,478,round(382*env.steps/cfg['max_steps']),5))
        status='PAUSED' if paused else env.outcome.upper()
        write(status,650,504,self.colors[0] if env.outcome!='wrong switch' else self.colors[1],self.medium)
        if env.done and not paused:
            write('R for next room' if mode=='Manual' else 'Next dive starts automatically',650,538,muted)
        rule='Hold the plate while the other agent activates the matching switch. Both must reach the hatch.'
        if relay: rule='Collect fuse [E]. Meet on opposite sides of the grille. Pass [F], install [E], then escape.'
        write(rule,26,582,ink)
        write('R  Next dive     P  Pause     N  Step     + / -  Speed     Esc  Quit',26,611,muted)
        if mode=='Manual':
            write('Tab  Select     Arrows  Move     E  Use     Space  Wait     0/1/2  Signal',26,634,brass)
