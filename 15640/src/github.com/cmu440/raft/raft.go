/* Student: Joao Monteiro (jmonteir)
 * Course: 15-640
 * Last Date Modified: 11/12/2022
 */

package raft

/**************************************************************************
// API
// ===
// This is an outline of the API that the raft implementation exposes.
//
// rf = NewPeer(...)
//   Create a new Raft server.
//
// rf.PutCommand(command interface{}) (index, term, isleader)
//   PutCommand agreement on a new log entry
//
// rf.GetState() (me, term, isLeader)
//   Ask a Raft peer for "me", its current term, and whether it thinks it
//   is a leader
//
// ApplyCommand
//   Each time a new entry is committed to the log, each Raft peer
//   should send an ApplyCommand to the service (e.g. tester) on the
//   same server, via the applyCh channel passed to NewPeer()
****************************************************************************/

import (
	"math/rand"
	"sync"
	"time"

	"github.com/cmu440/rpc"
)

type serverState int //possible states for a server in a term

const (
	Follower serverState = iota
	Candidate
	Leader
)

const milliSecBtwHeartBeats = 100 //milliseconds between heart beats
const lowElectionTOut = 400       //lower bound for the election interval timeout in ms
const upElectionTOut = 800        //upper bound for the election interval timeout in ms
const milliSecBtwApplyState = 300 //milliseconds between looking for commands to apply

// ============
// ApplyCommand
// ============
// As each Raft peer becomes aware that successive log entries are
// committed, the peer should send an ApplyCommand to the service (or
// tester) on the same server, via the applyCh passed to NewPeer()
type ApplyCommand struct {
	Index   int
	Command interface{}
}

// ========
// LogEntry
// ========
// Information stored in one entry of the log
type LogEntry struct {
	Index   int
	Term    int
	Command interface{}
}

// ===========
// Raft struct
// ===========
type Raft struct {
	mux               sync.Mutex       // Lock to protect shared access to this peer's state
	peers             []*rpc.ClientEnd // RPC end points of all peers
	me                int              // this peer's index into peers[]
	state             serverState      //state of the server in a term
	currentTerm       int              //current term
	votedFor          int              //who did this server voted for
	log               []LogEntry       //log entries; each entry contains command, term and index
	commitIndex       int              //index of highest log entry known to be committed
	lastApplied       int              // index of highets log entry applied to state machine
	nextIndex         map[int]int      //index of the next log entry to send to that server
	matchIndex        map[int]int      //index of highest log entry known to be replicated on server
	hbRecvdChan       chan bool        //channel to indicate a heart beat has been received
	sendHeartBeatChan chan bool        //channel to indicate that it is time to send a heart beat
	leaderNoMore      chan bool        //chanel to indicate server is no longer a leader, so stop sending HBs
	nVotesReceived    int              //number of votes received
	previousLeader    int              //hold information about who was leader prior to receive a log entry
	newLogEntryQueue  []LogEntry       //queue og new log entries. a heart beat sned them all entries in this queue
}

// ==========
// GetState()
// ==========
// Return "me", current term and whether this peer
// believes it is the leader
func (rf *Raft) GetState() (int, int, bool) {

	var me int
	var term int
	var isleader bool

	//collect server's state
	rf.mux.Lock()
	defer rf.mux.Unlock()
	me = rf.me
	term = rf.currentTerm
	isleader = (rf.state == Leader)

	return me, term, isleader
}

// ===================================
// RequestVote RPC arguments structure
// ===================================
type RequestVoteArgs struct {
	Term         int //candidate's term
	CandidateID  int //candidate requesting vote
	LastLogIndex int //index of candidate’s last log entry
	LastLogTerm  int //term of candidate’s last log entry
}

// ===============================
// RequestVote RPC reply structure
// ===============================
type RequestVoteReply struct {
	Term        int  //currentTerm, for candidate to update itself
	VoteGranted bool //true means candidate received vote
}

// =======================
// RequestVote RPC handler
// =======================
func (rf *Raft) RequestVote(args *RequestVoteArgs, reply *RequestVoteReply) {

	rf.hbRecvdChan <- true //heart beat received since there is a candidate out there

	rf.mux.Lock()
	defer rf.mux.Unlock()

	if args.Term < rf.currentTerm { //voter (i.e. me) has later term than candidate
		reply.VoteGranted = false
		reply.Term = rf.currentTerm
		return
	}

	if args.Term > rf.currentTerm { //candidate has later term than voter (i.e. me)
		rf.currentTerm = args.Term
		rf.votedFor = -1 // -1 has a meaning of NULL since golang does not allow nil for int type
		rf.state = Follower
		rf.nVotesReceived = 0
	}

	//checking if candidate's log is at least as up-to-date as mine
	candLogUpToDate := rf.isCandLogUpToDate(args.LastLogIndex, args.LastLogTerm,
		rf.log[len(rf.log)-1].Index,
		rf.log[len(rf.log)-1].Term)

	// if I haven't voted yet and this candidate has its log at least up-to-date as me, I will grant my vote
	if (rf.votedFor == -1 || rf.votedFor == args.CandidateID) && args.Term == rf.currentTerm && candLogUpToDate {
		reply.VoteGranted = true
		rf.votedFor = args.CandidateID
		rf.currentTerm = args.Term
		rf.state = Follower
		rf.nVotesReceived = 0
	} else { //not a good candidate, or I already voted
		reply.VoteGranted = false
	}
	reply.Term = rf.currentTerm
}

// =================
// isCandLogUpToDate
// =================
// This function checks whether the candidate's log is
// at least as up-to-date as receiver's log
func (rf *Raft) isCandLogUpToDate(candLastLogIndex, candLastLogTerm, voterLastLogIndex, voterLastLogTerm int) bool {

	var candLogUpToDate bool

	if candLastLogTerm > voterLastLogTerm { //candidate has log with later term
		candLogUpToDate = true
	} else if candLastLogTerm < voterLastLogTerm { //voter has log with later term
		candLogUpToDate = false
	} else { //both have their logs end with same term
		if candLastLogIndex >= voterLastLogIndex { //candidate has higher or equal index
			candLogUpToDate = true
		} else { //voter has higher index
			candLogUpToDate = false
		}
	}

	return candLogUpToDate
}

// ===============
// sendRequestVote
// ===============
//
// This function makes an RPC call requesting vote to server.
// It waits for the reply.
func (rf *Raft) sendRequestVote(server int, args *RequestVoteArgs) {

	var reply RequestVoteReply

	serverVoted := rf.peers[server].Call("Raft.RequestVote", args, &reply) //send request and wait for vote

	if serverVoted { //if server replied

		rf.mux.Lock()
		defer rf.mux.Unlock()

		if rf.state != Candidate { //if I am no longer a candidate the vote is meaningless
			return
		}

		//if voter's term is higher than mine I should become a follower
		if reply.Term > args.Term {
			rf.state = Follower
			rf.votedFor = -1
			rf.nVotesReceived = 0
			rf.currentTerm = reply.Term

			return
		}

		if reply.VoteGranted && reply.Term == rf.currentTerm { //only count vote if election still going on
			rf.nVotesReceived++

			if rf.nVotesReceived > len(rf.peers)/2 && rf.state != Leader { //if I win the election
				rf.state = Leader //change state to leader

				//initialize all nextIndex values to the index just after the last one in my log
				myNextIndex := len(rf.log)
				for s := 0; s < len(rf.peers); s++ {
					if s != rf.me {
						rf.nextIndex[s] = myNextIndex
						rf.matchIndex[s] = 0
					}
				}

				go rf.sendHeartBeatTimer() //declare my victory by sending a heart beat

				return
			}
		}
	}
}

// =====================================
// AppendEntries RPC arguments structure
// =====================================
type AppendEntriesArgs struct {
	Term         int        //leader's term
	LeaderId     int        //so follower can redirect clients
	PrevLogIndex int        //index of log entry immediately preceding new ones
	PrevLogTerm  int        //term of PrevLogIndex entry
	Entries      []LogEntry //log entries to store (empty for heartbeat)
	LeaderCommit int        //leader's commitIndex
}

// ==================================
// AppendEntries RPC reply structure
// ==================================
type AppendEntriesReply struct {
	Term    int  //receiver's term
	Success bool //true if follower contained entry matching prevLogIndex and prevLogTerm
}

// =========================
// AppendEntries RPC handler
// =========================
func (rf *Raft) AppendEntries(args *AppendEntriesArgs, reply *AppendEntriesReply) {
	rf.hbRecvdChan <- true
	rf.mux.Lock()
	defer rf.mux.Unlock()

	if args.Term >= rf.currentTerm {

		//if I am a leader but there is another leader with higher term than me, I should become a follower
		if rf.state == Leader && args.Term > rf.currentTerm {
			rf.leaderNoMore <- true            //shutdown my leader operations
			rf.log = rf.log[:rf.commitIndex+1] //get rid of my uncommited commands
		}

		//if leader changed since last append
		if rf.previousLeader == -1 || rf.previousLeader != args.LeaderId {
			rf.log = rf.log[:rf.commitIndex+1] //get rid of uncommited commands
			rf.previousLeader = args.LeaderId  //update previous leader
		}

		//change status to follower
		rf.state = Follower
		rf.nVotesReceived = 0
		rf.votedFor = -1
		rf.currentTerm = args.Term

		//update commit index
		if args.LeaderCommit > rf.commitIndex {
			rf.commitIndex = min(args.LeaderCommit, rf.log[len(rf.log)-1].Index)
		}

	} else { //I have higher term than leader. let it know
		reply.Success = false
		reply.Term = rf.currentTerm
		return
	}

	if len(args.Entries) == 0 { //heartbeat so no need to go over code below
		return
	}

	if len(rf.log) == 1 { //no client entries in the log of this server
		if args.PrevLogIndex > 0 { //leader has at least one client entry
			reply.Success = false //so the log of this server clearly doesn't contain an entry at prevLogIndex whose term matches prevLogTerm
		} else {
			reply.Success = true                     //both leader and this server have PrevLogIndex == 0
			rf.log = append(rf.log, args.Entries...) // append entries to the log

			//update commitIndex
			if args.LeaderCommit > rf.commitIndex {
				rf.commitIndex = min(args.LeaderCommit, rf.log[len(rf.log)-1].Index)
			}
		}
	} else {
		if args.PrevLogIndex > len(rf.log)-1 { //this server's last index is smaller than leader's PrevLogIndex.
			reply.Success = false //so clearly this server doesn't contain an entry at prevLogIndex whose term matches prevLogTerm
		} else {
			if rf.log[args.PrevLogIndex].Term != args.PrevLogTerm { //this server doesn't contain an entry at prevLogIndex whose term matches prevLogTerm
				reply.Success = false
			} else { //server contains an entry at prevLogIndex whose term matches prevLogTerm

				indexLastBlock := args.PrevLogIndex + len(args.Entries) + 1 //add new entry, but keep uncommitted commands to the right

				if indexLastBlock < len(rf.log) {
					rf.log = append(rf.log[:args.PrevLogIndex+1], append(args.Entries, rf.log[indexLastBlock:]...)...)
				} else {
					rf.log = append(rf.log[:args.PrevLogIndex+1], args.Entries...)
				}

				reply.Success = true

				//update commitIndex
				if args.LeaderCommit > rf.commitIndex {
					rf.commitIndex = min(args.LeaderCommit, rf.log[len(rf.log)-1].Index)
				}
			}
		}
	}

	reply.Term = rf.currentTerm
}

// ===============
// sendAppendEntries
// ===============
//
// This function makes an RPC call requesting servers to add entries in their logs.
// It waits for the reply.
func (rf *Raft) sendAppendEntries(server int, args *AppendEntriesArgs) {

	var reply AppendEntriesReply

	serverReplied := rf.peers[server].Call("Raft.AppendEntries", args, &reply) //send append request and wait for reply

	if serverReplied {
		rf.mux.Lock()
		defer rf.mux.Unlock()

		if reply.Term > args.Term { //server has higher term than me

			//if I am still a leader, I should shut down my leader operations
			if rf.state == Leader {
				rf.leaderNoMore <- true
				rf.log = rf.log[:rf.commitIndex+1] //get rid of uncommitted log entries
			}
			rf.state = Follower
			rf.votedFor = -1
			rf.nVotesReceived = 0
			rf.currentTerm = reply.Term

			return
		}

		if len(args.Entries) > 0 { //handle response from trying to append an entry
			if reply.Success { //sucess adding entries in the server
				rf.matchIndex[server] = args.PrevLogIndex + len(args.Entries)
				rf.nextIndex[server] += len(args.Entries)

				if rf.commitIndex+1 < len(rf.log) { //there is at least one log record uncommitted
					//check if there exists an N such that N > commitIndex, a majority of matchIndex[i] >= N
					for N := rf.commitIndex + 1; N < len(rf.log); N++ {
						nReplicated := 1
						for s := 0; s < len(rf.peers); s++ {
							if s != rf.me && rf.matchIndex[s] >= N {
								nReplicated++
							}
						}
						//if so and log[N].term == currentTerm than set commitIndex = N
						if nReplicated > len(rf.peers)/2 && rf.log[N].Term == rf.currentTerm {
							rf.commitIndex = N
						}
					}
				}

			} else { //follower did not contain entry matching prevLogIndex and prevLogTerm
				rf.nextIndex[server]-- //decrement nextIndex

				// and try again
				retryPrevLogIndex := args.PrevLogIndex - 1
				retryPrevLogTerm := rf.log[retryPrevLogIndex].Term
				retryEntries := append([]LogEntry{rf.log[args.PrevLogIndex]}, args.Entries...)
				retryLeaderCommit := rf.commitIndex

				argsAppendEntries := &AppendEntriesArgs{Term: rf.currentTerm,
					LeaderId:     rf.me,
					PrevLogIndex: retryPrevLogIndex,
					PrevLogTerm:  retryPrevLogTerm,
					Entries:      retryEntries,
					LeaderCommit: retryLeaderCommit}

				go rf.sendAppendEntries(server, argsAppendEntries)
			}
		}
	}
}

// ==========
// PutCommand
// ==========
//
// The service using Raft (e.g. a k/v server) wants to start
// agreement on the next command to be appended to Raft's log
//
// If this server is not the leader, return false
//
// Otherwise start the agreement and return immediately
//
// There is no guarantee that this command will ever be committed to
// the Raft log, since the leader may fail or lose an election
//
// The first return value is the index that the command will appear at
// if it is ever committed
//
// The second return value is the current term
//
// The third return value is true if this server believes it is
// the leader
func (rf *Raft) PutCommand(command interface{}) (int, int, bool) {

	rf.mux.Lock()
	defer rf.mux.Unlock()

	term := rf.currentTerm
	isLeader := (rf.state == Leader)

	if !isLeader { //I am not a leader
		return -1, term, false
	}

	//setup log entry
	index := rf.log[len(rf.log)-1].Index + 1 //first index is 1
	newLogEntry := LogEntry{Index: index, Term: term, Command: command}
	rf.log = append(rf.log, newLogEntry)

	//add new log entry to the queue, which will be processed in
	//the next heart beat
	rf.newLogEntryQueue = append(rf.newLogEntryQueue, newLogEntry)

	return index, term, isLeader
}

// ====
// Stop
// ====
//
// The tester calls Stop() when a Raft instance will not
// be needed again
//
// You are not required to do anything
// in Stop(), but it might be convenient to (for example)
// turn off debug output from this instance
//
func (rf *Raft) Stop() {
	// Your code here, if desired
}

// =======
// NewPeer
// =======
//
// The service or tester wants to create a Raft server
//
// The port numbers of all the Raft servers (including this one)
// are in peers[]
//
// This server's port is peers[me]
//
// All the servers' peers[] arrays have the same order
//
// applyCh
// =======
//
// applyCh is a channel on which the tester or service expects
// Raft to send ApplyCommand messages
//
// NewPeer() must return quickly, so it should start Goroutines
// for any long-running work
func NewPeer(peers []*rpc.ClientEnd, me int, applyCh chan ApplyCommand) *Raft {
	// initialization
	rf := &Raft{}
	rf.peers = peers
	rf.me = me
	rf.state = Follower
	rf.currentTerm = 0
	rf.votedFor = -1 //can't assign nil to int in go language. so -1 means null here
	rf.commitIndex = 0
	rf.lastApplied = 0
	rf.previousLeader = -1 //can't assign nil to int in go language. so -1 means null here
	rf.log = append(rf.log, LogEntry{Term: 0, Index: 0})
	rf.hbRecvdChan = make(chan bool)
	rf.sendHeartBeatChan = make(chan bool)
	rf.leaderNoMore = make(chan bool)
	rf.nextIndex = make(map[int]int)
	rf.matchIndex = make(map[int]int)

	go rf.serverActions()     //thread responsible for handling actions according to the state of the server
	go rf.applyState(applyCh) //thread responsible to apply commited commands

	return rf
}

// =============
// serverActions
// =============
// this function keeps track of election timer (follower, candidate),
// heart beat received (follower, candidate) and also heart beat/log append sent (leader)
func (rf *Raft) serverActions() {

	rand.Seed(time.Now().UnixNano())

	for {
		//set timer
		electionTimer := time.NewTimer(time.Duration(rand.Int()%(upElectionTOut-lowElectionTOut)+lowElectionTOut) * time.Millisecond)
		select {
		case <-rf.hbRecvdChan: //heart beat received, so reset timer
			electionTimer.Reset(time.Duration(rand.Int()%(upElectionTOut-lowElectionTOut)+lowElectionTOut) * time.Millisecond)

		case <-electionTimer.C: //timer is up. it is election time
			rf.electionRoutine()

		case <-rf.sendHeartBeatChan: //time to send a heart beat to my followers
			//resetting timer, so I do not start an election when I am still a leader!
			electionTimer.Reset(time.Duration(rand.Int()%(upElectionTOut-lowElectionTOut)+lowElectionTOut) * time.Millisecond)

			rf.mux.Lock()
			nPeers := len(rf.peers)

			prevLogIndex := 0
			prevLogTerm := 0

			if len(rf.newLogEntryQueue) > 0 { //if there are new log entries
				prevLogIndex = rf.newLogEntryQueue[0].Index - 1
				prevLogTerm = rf.log[prevLogIndex].Term
			}

			//put arguments in one struct
			argsHeartBeat := &AppendEntriesArgs{Term: rf.currentTerm, LeaderId: rf.me, PrevLogIndex: prevLogIndex,
				PrevLogTerm: prevLogTerm, Entries: rf.newLogEntryQueue, LeaderCommit: rf.commitIndex}

			latestState := rf.state
			rf.newLogEntryQueue = nil //empty queue
			rf.mux.Unlock()

			//if I am still a leader, send the heart beat/append log for each
			//of my followers
			if latestState == Leader {
				for s := 0; s < nPeers; s++ {
					if s != argsHeartBeat.LeaderId {
						go rf.sendAppendEntries(s, argsHeartBeat)
					}
				}
			}
		}
	}
}

// ===============
// electionRoutine
// ===============
// this function takes the actions needed for a server to request votes
// (i.e. increment term, server votes on itself, change state to candidate and call the function that
// makes the RPC RequestVote)
func (rf *Raft) electionRoutine() {

	rf.mux.Lock()

	if rf.state == Leader { //If I am already a leader no need to do the election routine
		rf.mux.Unlock()
		return
	}

	rf.currentTerm++      //increment current term
	rf.state = Candidate  //transition to Candidate state
	rf.nVotesReceived = 1 //vote on myself
	rf.votedFor = rf.me   //voted on myself

	var lastLogIndex, lastLogTerm int
	sizeLog := len(rf.log)

	//get information about candidate log
	if sizeLog > 0 {
		lastLogIndex = rf.log[sizeLog-1].Index
		lastLogTerm = rf.log[sizeLog-1].Term
	} else {
		lastLogIndex = -1
		lastLogTerm = 0
	}

	args := &RequestVoteArgs{Term: rf.currentTerm, CandidateID: rf.me,
		LastLogIndex: lastLogIndex, LastLogTerm: lastLogTerm}

	nPeers := len(rf.peers)
	rf.mux.Unlock()

	//go request vote for me
	for s := 0; s < nPeers; s++ {
		if s != args.CandidateID {
			go rf.sendRequestVote(s, args)
		}
	}

}

// ==================
// sendHeartBeatTimer
// ==================
// this function keeps track of the time to send a heart beat
func (rf *Raft) sendHeartBeatTimer() {

	for {
		select {
		case <-rf.leaderNoMore: //stop sending heart beats if I am no longer a leader
			return

		default:
			rf.sendHeartBeatChan <- true
			time.Sleep(milliSecBtwHeartBeats * time.Millisecond)
		}
	}
}

// ==========
// applyState
// ==========
// this function applies new commited commands
func (rf *Raft) applyState(applyCh chan ApplyCommand) {

	var entriesToApply []ApplyCommand
	var commandToApply ApplyCommand

	newEntriesToApply := false //assume no new entries to apply

	for {
		rf.mux.Lock()
		if len(rf.log) > 1 && rf.lastApplied < rf.commitIndex { //new committed entries since last time
			for i := rf.lastApplied + 1; i <= rf.commitIndex; i++ {
				commandToApply = ApplyCommand{Index: rf.log[i].Index, Command: rf.log[i].Command}
				entriesToApply = append(entriesToApply, commandToApply)
			}
			newEntriesToApply = true        //there are new commited entries
			rf.lastApplied = rf.commitIndex //update last applied
		}
		rf.mux.Unlock()

		if newEntriesToApply {
			for e := 0; e < len(entriesToApply); e++ { //apply new commited entries
				applyCh <- entriesToApply[e]
			}

			newEntriesToApply = false //set this flag to false again
			entriesToApply = nil
		}
		time.Sleep(milliSecBtwApplyState * time.Millisecond)
	}
}

// ====
// min
// ====
// this function computes the min of two integers
func min(x, y int) int {
	if x < y {
		return x
	} else {
		return y
	}
}
