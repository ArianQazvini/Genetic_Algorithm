# from game2 import Game as gm2
class Game:

    def __init__(self, levels, win_reward_allowed):
        # Get a list of strings as levels
        # Store level length to determine if a sequence of action passes all the steps

        self.levels = levels
        self.current_level_index = -1
        self.current_level_len = 0
        self.win_reward_allowed= win_reward_allowed

    def load_next_level(self):
        self.current_level_index += 1
        self.current_level_len = len(self.levels[self.current_level_index][1])

    def get_score(self, actions):
        # Get an action sequence and determine the steps taken/score
        # Return a tuple, the first one indicates if these actions result in victory
        # and the second one shows the steps taken
        current_level = self.levels[self.current_level_index][1]
        steps = 0

        WIN_REWARD = 5
        if(self.win_reward_allowed==False):
            WIN_REWARD=0
        MUSHROOM_REWARD = 2


        KILL_GUMPA_REWARD = 2
        DEATH_REWARD = -3
        CONSECUTIVE_JUMPS_PENALTY= -1
        JUMP_PENALTY=-0.3
        JUMP_BEFORE_END_REWARD = 1
        STEP_REWARD = 0.3


        score = 0
        max_steps= 0
        win = True
        # print(len(actions))
        for i in range(self.current_level_len):
            current_step = current_level[i]
            if (current_step == '_'):
                steps += 1
            elif (current_step == 'G' and actions[i - 1] == '1'):
                steps += 1
            elif (current_step == 'L' and actions[i - 1] == '2'):
                steps += 1
            elif(current_step =='M' and actions[i-1] !='1'):
                steps +=1
                score+=MUSHROOM_REWARD
            elif(current_step=='M' and actions[i-1]=='1'):
                steps+=1
            elif(current_step=='G' and (actions[i-2]=='1' and actions[i-1]=='0')):
                 steps+=1
                 score+=KILL_GUMPA_REWARD

            else:
                max_steps=max(max_steps,steps)
                steps=0
                win=False
            if(i==self.current_level_len-1 and actions[i-1]=='1'):
                score+=JUMP_BEFORE_END_REWARD
            if(actions[i-1]=='1'):
                score+=JUMP_PENALTY
        if(win==False):
            score+=DEATH_REWARD
        else:
            score+=WIN_REWARD
        score+=(max_steps*STEP_REWARD)
        return (win,score)


# test = gm2([('level2', '____G_____')])
# test.load_next_level()
# g = Game(['____G_____'],True)
# g.load_next_level()
# print(g.get_score("1010100100"))
# print(test.get_score("1010100100"))
# This outputs (False, 4)
# print(g.get_score("000000000"))
# print(g.get_score("000100200"))
