"""Stage two: a physical barrier admits the fuse before it admits the crew."""
from collections import deque
import numpy as np
from environment import EscapeRoom, IDS, MOVES

PASS = 6


class RelayRoom(EscapeRoom):
    moves = MOVES + ((0, 0),)

    def __init__(self, seed=0, communication=True):
        self.communication = communication
        self.reset(seed)

    def reset(self, seed=None, clue=None):
        rng = np.random.default_rng(seed)
        # Either side may have the supply, and either character may start there.
        supply, socket = (2, 6) if rng.integers(2) else (6, 2)
        starts = [[2, 3], [6, 3]]
        if rng.integers(2):
            starts.reverse()
        self.config = dict(width=9,height=7,starts=starts,max_steps=100,
                           walls=[[4,y] for y in range(1,6)],exit=[socket,4])
        self.fuse = (supply, int(rng.choice([1,5])))
        self.socket = (socket, int(rng.choice([1,5])))
        self.grille = (4,3)
        self.carrier = None
        self.event = 'Find the relay fuse'
        return super().reset(seed)

    def observations(self):
        layers=np.zeros((7,7,9),np.float32)
        layers[0,[0,-1],:]=1
        layers[0,:,[0,-1]]=1
        for x,y in self.config['walls']:
            layers[0,y,x]=1
        for layer,(x,y) in enumerate((self.fuse,self.socket,self.grille,self.config['exit']),1):
            if layer!=1 or (self.carrier is None and not self.unlocked):
                layers[layer,y,x]=1
        for i,(x,y) in enumerate(self.pos):
            if not self.escaped[i]:
                layers[5+i,y,x]=1
        observations={}
        for i,name in enumerate(IDS):
            state=[self.carrier==i,self.carrier==1-i,self.unlocked,self.escaped[i],
                   self.escaped[1-i],1-self.steps/self.config['max_steps']]
            msg=np.eye(3,dtype=np.float32)[self.messages[1-i]]
            observations[name]=np.concatenate((layers[[0,1,2,3,4,5+i,6-i]].ravel(),state,msg)).astype(np.float32)
        return observations

    def interact(self,pairs):
        if self.unlocked:
            return 0,False
        if self.carrier is None:
            for i,pos in enumerate(self.pos):
                if pos==self.fuse and pairs[i][0]==5:
                    self.carrier=i
                    self.event=f'{IDS[i]} picked up the fuse'
                    break
        elif pairs[self.carrier][0]==PASS and set(self.pos)=={(3,3),(5,3)}:
            self.carrier=1-self.carrier
            self.event=f'{IDS[self.carrier]} received the fuse'
        elif self.pos[self.carrier]==self.socket and pairs[self.carrier][0]==5:
            self.unlocked=True
            self.carrier=None
            self.config['walls'].remove(list(self.grille))
            self.event='Power restored'
            return 1,False
        return 0,False

    def reference_actions(self):
        """Scripted demonstration only. Policies receive observations(), never this method."""
        actions={}
        for i,name in enumerate(IDS):
            pos=self.pos[i]
            on_supply_side=(pos[0]<4)==(self.fuse[0]<4)
            at_window=(3 if pos[0]<4 else 5,3)
            use=4
            if self.unlocked:
                goal=tuple(self.config['exit'])
                # The socket-side agent clears the single doorway first.
                if on_supply_side and not self.escaped[1-i]:
                    goal=pos
            elif self.carrier==i:
                goal=at_window if on_supply_side else self.socket
                use=PASS if on_supply_side else 5
            elif on_supply_side and self.carrier is None:
                goal=self.fuse; use=5
            else:
                goal=at_window
            action=use if pos==goal else 4
            if pos!=goal and not self.escaped[i]:
                blocked=set(map(tuple,self.config['walls']))
                if not self.unlocked: blocked.add(tuple(self.config['exit']))
                if not self.escaped[1-i]: blocked.add(self.pos[1-i])
                queue=deque([(pos,None)]); seen={pos}
                while queue:
                    tile,first=queue.popleft()
                    if tile==goal:
                        action=first; break
                    for move,(dx,dy) in enumerate(MOVES[:4]):
                        cell=(tile[0]+dx,tile[1]+dy)
                        if 0<cell[0]<8 and 0<cell[1]<6 and cell not in seen and cell not in blocked:
                            seen.add(cell); queue.append((cell,move if first is None else first))
            actions[name]=(action,0)
        return actions
