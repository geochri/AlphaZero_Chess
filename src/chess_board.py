#!/usr/bin/env python

import numpy as np
import itertools
import copy

class board():
    def __init__(self):
        self.init_board = np.zeros([8,8]).astype(str)
        self.init_board[0,0] = "r"
        self.init_board[0,1] = "n"
        self.init_board[0,2] = "b"
        self.init_board[0,3] = "q"
        self.init_board[0,4] = "k"
        self.init_board[0,5] = "b"
        self.init_board[0,6] = "n"
        self.init_board[0,7] = "r"
        self.init_board[1,0:8] = "p"
        self.init_board[7,0] = "R"
        self.init_board[7,1] = "N"
        self.init_board[7,2] = "B"
        self.init_board[7,3] = "Q"
        self.init_board[7,4] = "K"
        self.init_board[7,5] = "B"
        self.init_board[7,6] = "N"
        self.init_board[7,7] = "R"
        self.init_board[6,0:8] = "P"
        self.init_board[self.init_board == "0.0"] = " "
        self.move_count = 0
        self.no_progress_count = 0
        self.repetitions_w = 0
        self.repetitions_b = 0
        self.move_history = None
        self.en_passant = -999; self.en_passant_move = 0 # returns j index of last en_passant pawn
        self.r1_move_count = 0 # black's queenside rook
        self.r2_move_count = 0 # black's kingside rook
        self.k_move_count = 0
        self.R1_move_count = 0 # white's queenside rook
        self.R2_move_count = 0 # white's kingside rook
        self.K_move_count = 0
        self.current_board = self.init_board
        self.en_passant_move_copy = None
        self.copy_board = None; self.en_passant_copy = None; self.r1_move_count_copy = None; self.r2_move_count_copy = None; 
        self.k_move_count_copy = None; self.R1_move_count_copy = None; self.R2_move_count_copy = None; self.K_move_count_copy = None
        self.player = 0 # current player's turn (0:white, 1:black)
        
    def move_rules_P(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        ## to calculate allowed moves for king
        threats = []
        if 0<=i-1<=7 and 0<=j+1<=7:
            threats.append((i-1,j+1))
        if 0<=i-1<=7 and 0<=j-1<=7:
            threats.append((i-1,j-1))
        #at initial position
        if i==6:
            if board_state[i-1,j]==" ":
                next_positions.append((i-1,j))
                if board_state[i-2,j]==" ":
                    next_positions.append((i-2,j))
        # en passant capture
        elif i==3 and self.en_passant!=-999:
            if j-1==self.en_passant and abs(self.en_passant_move-self.move_count) == 1:
                next_positions.append((i-1,j-1))
            elif j+1==self.en_passant and abs(self.en_passant_move-self.move_count) == 1:
                next_positions.append((i-1,j+1))
        if i in [1,2,3,4,5] and board_state[i-1,j]==" ":
            next_positions.append((i-1,j))          
        if j==0 and board_state[i-1,j+1] in ["r", "n", "b", "q", "k", "p"]:
            next_positions.append((i-1,j+1))
        elif j==7 and board_state[i-1,j-1] in ["r", "n", "b", "q", "k", "p"]:
            next_positions.append((i-1,j-1))
        elif j in [1,2,3,4,5,6]:
            if board_state[i-1,j+1] in ["r", "n", "b", "q", "k", "p"]:
                next_positions.append((i-1,j+1))
            if board_state[i-1,j-1] in ["r", "n", "b", "q", "k", "p"]:
                next_positions.append((i-1,j-1))
        return next_positions, threats
    
    def move_rules_p(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        ## to calculate allowed moves for king
        threats = []
        if 0<=i+1<=7 and 0<=j+1<=7:
            threats.append((i+1,j+1))
        if 0<=i+1<=7 and 0<=j-1<=7:
            threats.append((i+1,j-1))
        #at initial position
        if i==1:
            if board_state[i+1,j]==" ":
                next_positions.append((i+1,j))
                if board_state[i+2,j]==" ":
                    next_positions.append((i+2,j))
        # en passant capture
        elif i==4 and self.en_passant!=-999:
            if j-1==self.en_passant and abs(self.en_passant_move-self.move_count) == 1:
                next_positions.append((i+1,j-1))
            elif j+1==self.en_passant and abs(self.en_passant_move-self.move_count) == 1:
                next_positions.append((i+1,j+1))
        if i in [2,3,4,5,6] and board_state[i+1,j]==" ":
            next_positions.append((i+1,j))          
        if j==0 and board_state[i+1,j+1] in ["R", "N", "B", "Q", "K", "P"]:
            next_positions.append((i+1,j+1))
        elif j==7 and board_state[i+1,j-1] in ["R", "N", "B", "Q", "K", "P"]:
            next_positions.append((i+1,j-1))
        elif j in [1,2,3,4,5,6]:
            if board_state[i+1,j+1] in ["R", "N", "B", "Q", "K", "P"]:
                next_positions.append((i+1,j+1))
            if board_state[i+1,j-1] in ["R", "N", "B", "Q", "K", "P"]:
                next_positions.append((i+1,j-1))
        return next_positions, threats
    
    def move_rules_r(self,current_position):
        i, j = current_position
        board_state = self.current_board
        next_positions = []; a=i
        while a!=0:
            if board_state[a-1,j]!=" ":
                if board_state[a-1,j] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a-1,j))
                break
            next_positions.append((a-1,j))
            a-=1
        a=i
        while a!=7:
            if board_state[a+1,j]!=" ":
                if board_state[a+1,j] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a+1,j))
                break
            next_positions.append((a+1,j))
            a+=1
        a=j
        while a!=7:
            if board_state[i,a+1]!=" ":
                if board_state[i,a+1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((i,a+1))
                break
            next_positions.append((i,a+1))
            a+=1
        a=j
        while a!=0:
            if board_state[i,a-1]!=" ":
                if board_state[i,a-1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((i,a-1))
                break
            next_positions.append((i,a-1))
            a-=1
        return next_positions
    
    def move_rules_R(self,current_position):
        i, j = current_position
        board_state = self.current_board
        next_positions = []; a=i
        while a!=0:
            if board_state[a-1,j]!=" ":
                if board_state[a-1,j] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a-1,j))
                break
            next_positions.append((a-1,j))
            a-=1
        a=i
        while a!=7:
            if board_state[a+1,j]!=" ":
                if board_state[a+1,j] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a+1,j))
                break
            next_positions.append((a+1,j))
            a+=1
        a=j
        while a!=7:
            if board_state[i,a+1]!=" ":
                if board_state[i,a+1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((i,a+1))
                break
            next_positions.append((i,a+1))
            a+=1
        a=j
        while a!=0:
            if board_state[i,a-1]!=" ":
                if board_state[i,a-1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((i,a-1))
                break
            next_positions.append((i,a-1))
            a-=1
        return next_positions
    
    def move_rules_n(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        for a,b in [(i+2,j-1),(i+2,j+1),(i+1,j-2),(i-1,j-2),(i-2,j+1),(i-2,j-1),(i-1,j+2),(i+1,j+2)]:
            if 0<=a<=7 and 0<=b<=7:
                if board_state[a,b] in ["R", "N", "B", "Q", "K", "P", " "]:
                    next_positions.append((a,b))
        return next_positions
    
    def move_rules_N(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        for a,b in [(i+2,j-1),(i+2,j+1),(i+1,j-2),(i-1,j-2),(i-2,j+1),(i-2,j-1),(i-1,j+2),(i+1,j+2)]:
            if 0<=a<=7 and 0<=b<=7:
                if board_state[a,b] in ["r", "n", "b", "q", "k", "p", " "]:
                    next_positions.append((a,b))
        return next_positions
    
    def move_rules_b(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        a=i;b=j
        while a!=0 and b!=0:
            if board_state[a-1,b-1]!=" ":
                if board_state[a-1,b-1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a-1,b-1))
                break
            next_positions.append((a-1,b-1))
            a-=1;b-=1
        a=i;b=j
        while a!=7 and b!=7:
            if board_state[a+1,b+1]!=" ":
                if board_state[a+1,b+1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a+1,b+1))
                break
            next_positions.append((a+1,b+1))
            a+=1;b+=1
        a=i;b=j
        while a!=0 and b!=7:
            if board_state[a-1,b+1]!=" ":
                if board_state[a-1,b+1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a-1,b+1))
                break
            next_positions.append((a-1,b+1))
            a-=1;b+=1
        a=i;b=j
        while a!=7 and b!=0:
            if board_state[a+1,b-1]!=" ":
                if board_state[a+1,b-1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a+1,b-1))
                break
            next_positions.append((a+1,b-1))
            a+=1;b-=1
        return next_positions
    
    def move_rules_B(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        a=i;b=j
        while a!=0 and b!=0:
            if board_state[a-1,b-1]!=" ":
                if board_state[a-1,b-1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a-1,b-1))
                break
            next_positions.append((a-1,b-1))
            a-=1;b-=1
        a=i;b=j
        while a!=7 and b!=7:
            if board_state[a+1,b+1]!=" ":
                if board_state[a+1,b+1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a+1,b+1))
                break
            next_positions.append((a+1,b+1))
            a+=1;b+=1
        a=i;b=j
        while a!=0 and b!=7:
            if board_state[a-1,b+1]!=" ":
                if board_state[a-1,b+1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a-1,b+1))
                break
            next_positions.append((a-1,b+1))
            a-=1;b+=1
        a=i;b=j
        while a!=7 and b!=0:
            if board_state[a+1,b-1]!=" ":
                if board_state[a+1,b-1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a+1,b-1))
                break
            next_positions.append((a+1,b-1))
            a+=1;b-=1
        return next_positions
    
    def move_rules_q(self,current_position):
        i, j = current_position
        board_state = self.current_board
        next_positions = [];a=i
        #bishop moves
        while a!=0:
            if board_state[a-1,j]!=" ":
                if board_state[a-1,j] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a-1,j))
                break
            next_positions.append((a-1,j))
            a-=1
        a=i
        while a!=7:
            if board_state[a+1,j]!=" ":
                if board_state[a+1,j] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a+1,j))
                break
            next_positions.append((a+1,j))
            a+=1
        a=j
        while a!=7:
            if board_state[i,a+1]!=" ":
                if board_state[i,a+1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((i,a+1))
                break
            next_positions.append((i,a+1))
            a+=1
        a=j
        while a!=0:
            if board_state[i,a-1]!=" ":
                if board_state[i,a-1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((i,a-1))
                break
            next_positions.append((i,a-1))
            a-=1
        #rook moves
        a=i;b=j
        while a!=0 and b!=0:
            if board_state[a-1,b-1]!=" ":
                if board_state[a-1,b-1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a-1,b-1))
                break
            next_positions.append((a-1,b-1))
            a-=1;b-=1
        a=i;b=j
        while a!=7 and b!=7:
            if board_state[a+1,b+1]!=" ":
                if board_state[a+1,b+1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a+1,b+1))
                break
            next_positions.append((a+1,b+1))
            a+=1;b+=1
        a=i;b=j
        while a!=0 and b!=7:
            if board_state[a-1,b+1]!=" ":
                if board_state[a-1,b+1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a-1,b+1))
                break
            next_positions.append((a-1,b+1))
            a-=1;b+=1
        a=i;b=j
        while a!=7 and b!=0:
            if board_state[a+1,b-1]!=" ":
                if board_state[a+1,b-1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a+1,b-1))
                break
            next_positions.append((a+1,b-1))
            a+=1;b-=1
        return next_positions
    
    def move_rules_Q(self,current_position):
        i, j = current_position
        board_state = self.current_board
        next_positions = [];a=i
        #bishop moves
        while a!=0:
            if board_state[a-1,j]!=" ":
                if board_state[a-1,j] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a-1,j))
                break
            next_positions.append((a-1,j))
            a-=1
        a=i
        while a!=7:
            if board_state[a+1,j]!=" ":
                if board_state[a+1,j] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a+1,j))
                break
            next_positions.append((a+1,j))
            a+=1
        a=j
        while a!=7:
            if board_state[i,a+1]!=" ":
                if board_state[i,a+1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((i,a+1))
                break
            next_positions.append((i,a+1))
            a+=1
        a=j
        while a!=0:
            if board_state[i,a-1]!=" ":
                if board_state[i,a-1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((i,a-1))
                break
            next_positions.append((i,a-1))
            a-=1
        #rook moves
        a=i;b=j
        while a!=0 and b!=0:
            if board_state[a-1,b-1]!=" ":
                if board_state[a-1,b-1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a-1,b-1))
                break
            next_positions.append((a-1,b-1))
            a-=1;b-=1
        a=i;b=j
        while a!=7 and b!=7:
            if board_state[a+1,b+1]!=" ":
                if board_state[a+1,b+1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a+1,b+1))
                break
            next_positions.append((a+1,b+1))
            a+=1;b+=1
        a=i;b=j
        while a!=0 and b!=7:
            if board_state[a-1,b+1]!=" ":
                if board_state[a-1,b+1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a-1,b+1))
                break
            next_positions.append((a-1,b+1))
            a-=1;b+=1
        a=i;b=j
        while a!=7 and b!=0:
            if board_state[a+1,b-1]!=" ":
                if board_state[a+1,b-1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a+1,b-1))
                break
            next_positions.append((a+1,b-1))
            a+=1;b-=1
        return next_positions
    
    #does not include king, castling
    def possible_W_moves(self, threats=False):
        board_state = self.current_board
        rooks = {}; knights = {}; bishops = {}; queens = {}; pawns = {};
        i,j = np.where(board_state=="R")
        for rook in zip(i,j):
            rooks[tuple(rook)] = self.move_rules_R(rook)
        i,j = np.where(board_state=="N")
        for knight in zip(i,j):
            knights[tuple(knight)] = self.move_rules_N(knight)
        i,j = np.where(board_state=="B")
        for bishop in zip(i,j):
            bishops[tuple(bishop)] = self.move_rules_B(bishop)
        i,j = np.where(board_state=="Q")
        for queen in zip(i,j):
            queens[tuple(queen)] = self.move_rules_Q(queen)
        i,j = np.where(board_state=="P")
        for pawn in zip(i,j):
            if threats==False:
                pawns[tuple(pawn)],_ = self.move_rules_P(pawn)
            else:
                _,pawns[tuple(pawn)] = self.move_rules_P(pawn)
        c_dict = {"R": rooks, "N": knights, "B": bishops, "Q": queens, "P": pawns}
        c_list = []
        c_list.extend(list(itertools.chain(*list(rooks.values())))); c_list.extend(list(itertools.chain(*list(knights.values())))); 
        c_list.extend(list(itertools.chain(*list(bishops.values())))); c_list.extend(list(itertools.chain(*list(queens.values()))))
        c_list.extend(list(itertools.chain(*list(pawns.values()))))
        return c_list, c_dict
        
    def move_rules_k(self):
        current_position = np.where(self.current_board=="k")
        i, j = current_position; i,j = i[0],j[0]
        next_positions = []
        c_list, _ = self.possible_W_moves(threats=True)
        for a,b in [(i+1,j),(i-1,j),(i,j+1),(i,j-1),(i+1,j+1),(i-1,j-1),(i+1,j-1),(i-1,j+1)]:
            if 0<=a<=7 and 0<=b<=7:
                if self.current_board[a,b] in [" ","Q","B","N","P","R"] and (a,b) not in c_list:
                    next_positions.append((a,b))
        if self.castle("queenside") == True and self.check_status() == False:
            next_positions.append((0,2))
        if self.castle("kingside") == True and self.check_status() == False:
            next_positions.append((0,6))
        return next_positions
    
        #does not include king, castling
    def possible_B_moves(self,threats=False):
        rooks = {}; knights = {}; bishops = {}; queens = {}; pawns = {};
        board_state = self.current_board
        i,j = np.where(board_state=="r")
        for rook in zip(i,j):
            rooks[tuple(rook)] = self.move_rules_r(rook)
        i,j = np.where(board_state=="n")
        for knight in zip(i,j):
            knights[tuple(knight)] = self.move_rules_n(knight)
        i,j = np.where(board_state=="b")
        for bishop in zip(i,j):
            bishops[tuple(bishop)] = self.move_rules_b(bishop)
        i,j = np.where(board_state=="q")
        for queen in zip(i,j):
            queens[tuple(queen)] = self.move_rules_q(queen)
        i,j = np.where(board_state=="p")
        for pawn in zip(i,j):
            if threats==False:
                pawns[tuple(pawn)],_ = self.move_rules_p(pawn)
            else:
                _,pawns[tuple(pawn)] = self.move_rules_p(pawn)
        c_dict = {"r": rooks, "n": knights, "b": bishops, "q": queens, "p": pawns}
        c_list = []
        c_list.extend(list(itertools.chain(*list(rooks.values())))); c_list.extend(list(itertools.chain(*list(knights.values())))); 
        c_list.extend(list(itertools.chain(*list(bishops.values())))); c_list.extend(list(itertools.chain(*list(queens.values()))))
        c_list.extend(list(itertools.chain(*list(pawns.values()))))
        return c_list, c_dict
        
    def move_rules_K(self):
        current_position = np.where(self.current_board=="K")
        i, j = current_position; i,j = i[0],j[0]
        next_positions = []
        c_list, _ = self.possible_B_moves(threats=True)
        for a,b in [(i+1,j),(i-1,j),(i,j+1),(i,j-1),(i+1,j+1),(i-1,j-1),(i+1,j-1),(i-1,j+1)]:
            if 0<=a<=7 and 0<=b<=7:
                if self.current_board[a,b] in [" ","q","b","n","p","r"] and (a,b) not in c_list:
                    next_positions.append((a,b))
        if self.castle("queenside") == True and self.check_status() == False:
            next_positions.append((7,2))
        if self.castle("kingside") == True and self.check_status() == False:
            next_positions.append((7,6))
        return next_positions
    
    def move_piece(self,initial_position,final_position,promoted_piece="Q"):
        if self.player == 0:
            promoted = False
            i, j = initial_position
            piece = self.current_board[i,j]
            self.current_board[i,j] = " "
            i, j = final_position
            if piece == "R" and initial_position == (7,0):
                self.R1_move_count += 1
            if piece == "R" and initial_position == (7,7):
                self.R2_move_count += 1
            if piece == "K":
                self.K_move_count += 1
            x, y = initial_position
            if piece == "P":
                if abs(x-i) > 1:
                    self.en_passant = j; self.en_passant_move = self.move_count
                if abs(y-j) == 1 and self.current_board[i,j] == " ": # En passant capture
                    self.current_board[i+1,j] = " "
                if i == 0 and promoted_piece in ["R","B","N","Q"]:
                    self.current_board[i,j] = promoted_piece
                    promoted = True
            if promoted == False:
                self.current_board[i,j] = piece
            self.player = 1
            self.move_count += 1
    
        elif self.player == 1:
            promoted = False
            i, j = initial_position
            piece = self.current_board[i,j]
            self.current_board[i,j] = " "
            i, j = final_position
            if piece == "r" and initial_position == (0,0):
                self.r1_move_count += 1
            if piece == "r" and initial_position == (0,7):
                self.r2_move_count += 1
            if piece == "k":
                self.k_move_count += 1
            x, y = initial_position
            if piece == "p":
                if abs(x-i) > 1:
                    self.en_passant = j; self.en_passant_move = self.move_count
                if abs(y-j) == 1 and self.current_board[i,j] == " ": # En passant capture
                    self.current_board[i-1,j] = " "
                if i == 7 and promoted_piece in ["r","b","n","q"]:
                    self.current_board[i,j] = promoted_piece
                    promoted = True
            if promoted == False:
                self.current_board[i,j] = piece
            self.player = 0
            self.move_count += 1

        else:
            print("Invalid move: ",initial_position,final_position,promoted_piece)
        
        
    ## player = "w" or "b", side="queenside" or "kingside"
    def castle(self,side,inplace=False):
        if self.player == 0 and self.K_move_count == 0:
            if side == "queenside" and self.R1_move_count == 0 and self.current_board[7,1] == " " and self.current_board[7,2] == " "\
                and self.current_board[7,3] == " ":
                if inplace == True:
                    self.current_board[7,0] = " "; self.current_board[7,3] = "R"
                    self.current_board[7,4] = " "; self.current_board[7,2] = "K"
                    self.K_move_count += 1
                    self.player = 1
                return True
            elif side == "kingside" and self.R2_move_count == 0 and self.current_board[7,5] == " " and self.current_board[7,6] == " ":
                if inplace == True:
                    self.current_board[7,7] = " "; self.current_board[7,5] = "R"
                    self.current_board[7,4] = " "; self.current_board[7,6] = "K"
                    self.K_move_count += 1
                    self.player = 1
                return True
        if self.player == 1 and self.k_move_count == 0:
            if side == "queenside" and self.r1_move_count == 0 and self.current_board[0,1] == " " and self.current_board[0,2] == " "\
                and self.current_board[0,3] == " ":
                if inplace == True:
                    self.current_board[0,0] = " "; self.current_board[0,3] = "r"
                    self.current_board[0,4] = " "; self.current_board[0,2] = "k"
                    self.k_move_count += 1
                    self.player = 0
                return True
            elif side == "kingside" and self.r2_move_count == 0 and self.current_board[0,5] == " " and self.current_board[0,6] == " ":
                if inplace == True:
                    self.current_board[0,7] = " "; self.current_board[0,5] = "r"
                    self.current_board[0,4] = " "; self.current_board[0,6] = "k"
                    self.k_move_count += 1
                    self.player = 0
                return True
        return False
    
    ## Check if current player's king is in check
    def check_status(self):
        if self.player == 0:
            c_list,_ = self.possible_B_moves(threats=True)
            king_position = np.where(self.current_board=="K")
            i, j = king_position
            if (i,j) in c_list:
                return True
        elif self.player == 1:
            c_list,_ = self.possible_W_moves(threats=True)
            king_position = np.where(self.current_board=="k")
            i, j = king_position
            if (i,j) in c_list:
                return True
        return False
    
    def in_check_possible_moves(self):
        self.copy_board = copy.deepcopy(self.current_board); self.move_count_copy = self.move_count # backup board state
        self.en_passant_copy = copy.deepcopy(self.en_passant); self.r1_move_count_copy = copy.deepcopy(self.r1_move_count); 
        self.r2_move_count_copy = copy.deepcopy(self.r2_move_count); self.en_passant_move_copy = copy.deepcopy(self.en_passant_move)
        self.k_move_count_copy = copy.deepcopy(self.k_move_count); self.R1_move_count_copy = copy.deepcopy(self.R1_move_count); 
        self.R2_move_count_copy = copy.deepcopy(self.R2_move_count)
        self.K_move_count_copy = copy.deepcopy(self.K_move_count)
        if self.player == 0:
            possible_moves = []
            _, c_dict = self.possible_W_moves()
            current_position = np.where(self.current_board=="K")
            i, j = current_position; i,j = i[0],j[0]
            c_dict["K"] = {(i,j):self.move_rules_K()}
            for key in c_dict.keys():
                for initial_pos in c_dict[key].keys():
                    for final_pos in c_dict[key][initial_pos]:
                        self.move_piece(initial_pos,final_pos)
                        self.player = 0 # reset board
                        if self.check_status() == False:
                            possible_moves.append([initial_pos, final_pos])
                        self.current_board = copy.deepcopy(self.copy_board);
                        self.en_passant = copy.deepcopy(self.en_passant_copy); self.en_passant_move = copy.deepcopy(self.en_passant_move_copy)
                        self.R1_move_count = copy.deepcopy(self.R1_move_count_copy); self.R2_move_count = copy.deepcopy(self.R2_move_count_copy)
                        self.K_move_count = copy.deepcopy(self.K_move_count_copy); self.move_count = self.move_count_copy
            return possible_moves
        if self.player == 1:
            possible_moves = []
            _, c_dict = self.possible_B_moves()
            current_position = np.where(self.current_board=="k")
            i, j = current_position; i,j = i[0],j[0]
            c_dict["k"] = {(i,j):self.move_rules_k()}
            for key in c_dict.keys():
                for initial_pos in c_dict[key].keys():
                    for final_pos in c_dict[key][initial_pos]:
                        self.move_piece(initial_pos,final_pos)
                        self.player = 1 # reset board
                        if self.check_status() == False:
                            possible_moves.append([initial_pos, final_pos])
                        self.current_board = copy.deepcopy(self.copy_board);
                        self.en_passant = copy.deepcopy(self.en_passant_copy); self.en_passant_move = copy.deepcopy(self.en_passant_move_copy)
                        self.r1_move_count = copy.deepcopy(self.r1_move_count_copy); self.r2_move_count = copy.deepcopy(self.r2_move_count_copy)
                        self.k_move_count = copy.deepcopy(self.k_move_count_copy); self.move_count = self.move_count_copy
            return possible_moves
    
    def actions(self): # returns all possible actions while not in check: initial_pos,final_pos,underpromote
        acts = []
        if self.player == 0:
            _,c_dict = self.possible_W_moves() # all non-king moves except castling
            current_position = np.where(self.current_board=="K")
            i, j = current_position; i,j = i[0],j[0]
            c_dict["K"] = {(i,j):self.move_rules_K()} # all king moves
            for key in c_dict.keys():
                for initial_pos in c_dict[key].keys():
                    for final_pos in c_dict[key][initial_pos]:
                        if key in ["P","p"] and final_pos[0] in [0,7]:
                            for p in ["queen","rook","knight","bishop"]:
                                acts.append([initial_pos,final_pos,p])
                        else:
                            acts.append([initial_pos,final_pos,None])
            actss = []
            for act in acts:  ## after move, check that its not check ownself, else illegal move
                i,f,p = act; b = copy.deepcopy(self)
                b.move_piece(i,f,p)
                b.player = 0
                if b.check_status() == False:
                    actss.append(act)
            return actss
        if self.player == 1:
            _,c_dict = self.possible_B_moves() # all non-king moves except castling
            current_position = np.where(self.current_board=="k")
            i, j = current_position; i,j = i[0],j[0]
            c_dict["k"] = {(i,j):self.move_rules_k()} # all king moves
            for key in c_dict.keys():
                for initial_pos in c_dict[key].keys():
                    for final_pos in c_dict[key][initial_pos]:
                        if key in ["P","p"] and final_pos[0] in [0,7]:
                            for p in ["queen","rook","knight","bishop"]:
                                acts.append([initial_pos,final_pos,p])
                        else:
                            acts.append([initial_pos,final_pos,None])
            actss = []
            for act in acts:  ## after move, check that its not check ownself, else illegal move
                i,f,p = act; b = copy.deepcopy(self)
                b.move_piece(i,f,p)
                b.player = 1
                if b.check_status() == False:
                    actss.append(act)
            return actss
        