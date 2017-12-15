#Tristan McRae
#AA 228 - Decision Making Under Uncertainty
#Final Project
#Due 12/10/16

using DataFrames

numDrives = 100 #edit based on number of trials you want to try
numRuns = 10 #number of times to run MCTS for each action
terminalTime = 100 #edit based on max # of steps each trial should take
count = 0
currentTime = 0
safeDistance = 5
maxPedestrianDistance = 10 #distance a pedestrian can go in 1 second
speedConversion = 5 #how many meters a velocity of 1 goes in 1 second
k = .5           #parameter for choosing action
alpha = .85        #parameter for choosing action
c = 100              #exploration constant
gamma = .99         #discount factor

columnNames = ["s", "a", "Q", "N", "r"]
pairs = DataFrame(v1 = [0], v2 = [0], v3 = [0.0], v4 = [0], v5 = [0.0])
names!(pairs.colindex, map(parse, columnNames))

#MCTS-------------------------------------------------------------------------

function reward(probability, crash, terminal)
  #println("reward")
  #gives reward for actions based on probability
  if (crash == 1)
    r = 0
  elseif (terminal == 1)
    r = -Inf
  else
    r = log(probability)
  end
  return r
end

function getAsAndNs(state, pairs)
  As = 0
  Ns = 0
  exists = false
  numPairs = size(pairs, 1)

  for i = 1:numPairs
    if (pairs[i, Symbol("s")] == state)
      if (pairs[i, Symbol("a")] != 0)  #doesn't count initialization action as an action
        As = As + 1
        Ns = Ns + pairs[i, Symbol("N")]
      end
      exists = true
    end
  end

  return [As, Ns, exists]
end

function exploreOrExploit(As, Ns)
  #first widening criteria
  if (Ns == 0)
    explore = true
  else
    explore = As < (k*Ns^alpha)
  end
  return explore
end

function exploit(state, Ns, pairs)
  #println("exploit")
  #chooses action to maximize value function
  action = 0
  maxValue = -Inf
  numPairs = size(pairs, 1)

  for i = 1:numPairs
    if (pairs[i, Symbol("s")] == state)
      Qsa = pairs[i, Symbol("Q")]
      Nsa = pairs[i, Symbol("N")]
      a = pairs[i, Symbol("a")]
      value = Qsa + c*sqrt(log(Ns)/Nsa)
      if (value >= maxValue && a != 0)
        maxValue = value
        action = a
      end
    end
  end

  return action
end

function getQandN(state, action, pairs)
  #println("getQandN")
  row = getRow(state, action, pairs)
  if (row != 0)
    Q = pairs[row, Symbol("Q")]
    N = pairs[row, Symbol("N")]
  else
    Q = 0
    N = -1
  end
  return [Q, N, row]
end

function tableReward(state, action, pairs)
  #println("tableReward")
  row = getRow(state, action, pairs)
  if (row != 0)
    r = pairs[row, Symbol("r")]
  else
    r = -Inf
  end
  return [r]
end

function addPair(state, action, q, r, pairs)
  push!(pairs, @data([state, action, q, 0, r])) #initialize with N = 0
  return pairs
end

function getRow(state, action, pairs)
  #println("getRow")
  #gets the row, if any, that the state action pair appears in
  numPairs = size(pairs, 1)
  row = 0

  for i = 1:numPairs
    if (pairs[i, Symbol("s")] == state && pairs[i, Symbol("a")] == action)
      row = i
      break
    end
  end

  return convert(Int64, row)
end

function updateQ(state, action, q, pairs)
  #println("updateQ")
  #updates Q and N when there is a new observation for a state action pair
  QandN = getQandN(state, action, pairs)
  oldQ = QandN[1]
  newN = QandN[2]
  row = convert(Int64, QandN[3])
  newQ = oldQ + ((q - oldQ) / newN)
  if (newQ == [-Inf]) newQ = -Inf end
  if (newQ[1] >= 0 || newQ[1]<0)  #Only update q if new Q is a number
    pairs[row, Symbol("Q")] = newQ[1]
  end
  return pairs
end

function incrementN(state, action, pairs)
  #println("incrementN")
  #increases N for given state action pair by 1
  QandN = getQandN(state, action, pairs)
  newN = QandN[2] + 1
  row = convert(Int64, QandN[3])
  pairs[row, Symbol("N")] = newN
  return pairs
end

function emptyPairs()
  #println("emptyPairs")
  #makes the table of state action pairs empty
  newDF = DataFrame(v1 = [0], v2 = [0], v3 = [0], v4 = [0], v5 = [0])
  names!(newDF.colindex, map(parse, columnNames))
  return newDF
end

function getAction()
  #println("getAction")
  #randomly selects a seed
    maxSeed =  2^50
    seed = rand(0 : maxSeed)
  return seed
end

function maxQaction(state, pairs)
  #println("maxQaction")
  numPairs = size(pairs, 1)
  maxQ = -Inf
  action = 0

  for i = 1:numPairs
    if (pairs[i, Symbol("s")] == state && pairs[i, Symbol("Q")] >= maxQ && pairs[i, Symbol("a")] !=0)
      maxQ = pairs[i, Symbol("Q")]
      action = pairs[i, Symbol("a")]
    end
  end

  return action
end

#Black-Box simulator----------------------------------------------------------

function pedestrians(state, seed)
  #println("pedestrians")
  srand(seed)
  xmag = trunc(rand() * 6)
  xdir = rand()
  if (xdir>.5) negativeX = 0 else negativeX = 1 end
  ymag = trunc(rand() * 6)
  ydir = rand()
  if (ydir>.5) negativeY = 0 else negativeY = 1 end
  deltaX = xmag * (-1)^negativeX
  deltaY = ymag * (-1)^negativeY
  vxy = stateToValues(state)
  v = vxy[1]
  x = vxy[2]
  y = vxy[3]
  x = newX(x, deltaX)
  y = newY(y, deltaY)
  newState = stateFromValues(v, x, y)
end

function newX(x, dx)
  #println("newX")
  if (dx>0) newX = min(x+dx,9) else newX = max(x+dx, 0) end
  return newX
end

function newY(y, dy)
  #println("newY")
  if (dy>0) newY = min(y+dy,99) else newY = max(y+dy, -99) end
  return newY
end

function carAction(state)
  #println("carAction")
  status = stateToValues(state)
  v = status[1]
  x = status[2]
  y = status[3]
  r = sqrt(x^2 + y^2) #makes decision to brake based on if pedestrian is within a certain range
  if (y > 0)
    if (v < 4)
      vNew = v + 1
    else
      vNew = v
    end
  else
    if (r < safeDistance)
      vNew = v - 1
    else
      vNew = v
    end
  end
  xNew = x
  yNew = div((y + (v+vNew)/2 * speedConversion),1)
  yNew = max(min(yNew, 99),-99)
  #print("yNew=")
  #println(yNew)
  newState = stateFromValues(vNew, xNew, yNew)
  return newState
end

function stateFromValues(v, x, y)
  #println("stateFromValues")
  v = round(v)
  x = round(x)
  y = round(y)
  state = abs(v) + 10 * abs(x) + 100 * abs(y)
  if (y >= 0) state = state + 10000 end
  return state
end

function stateToValues(state)
  #println("stateToValues")
  v = state % 10
  x = ((state - v) % 100 ) / 10
  yAbs = ((state - v - 10*x) % 10000) / 100
  positive = div(state, 10000)
  y = yAbs * (-1)^(1-positive)
  return [v, x, y]
end

function initialize()
  #println("initialize")
  #restarts process
  v = 4
  x = 2
  y = -50
  state = stateFromValues(v, x, y)
  return state
end

function nextStep(initialState, seed)
  #println("nextStep")
  #use carSensors and carActions to find out what happens next
  initialStateInfo = stateToValues(initialState)

  intermediateState = pedestrians(initialState, seed)
  intermediateStateInfo = stateToValues(intermediateState)
  y1 = initialStateInfo[3]
  p = probability(initialState, intermediateState)
  nextState = carAction(intermediateState)
  nextStateInfo = stateToValues(nextState)
  x = nextStateInfo[2]
  y2 = nextStateInfo[3]
  crash = isCrash(x, y1, y2)
  terminal = crash||hazardPassed(y2)

  return [p, crash, nextState, terminal]
end

function probability(state1, state2)
  #println("probability")
  vxy1 = stateToValues(state1)
  vxy2 = stateToValues(state2)
  x1 = vxy1[2]
  y1 = vxy1[3]
  x2 = vxy2[2]
  y2 = vxy2[3]

  if (x2 == 0)
    #stuck in road
    if (x1 == 0)
      px = 7/12
    else
      px = 0.5 * (6-x1)/6
    end
  elseif (x2 == 9)
    #stuck on far side
    if (x1 == 9)
      px = 7/12
    else
      px = 0.5 * (x1-3)/6
    end
  elseif (x1 == x2)
    #regular transition
    px = 1/6
  else
    px = 1/12
  end

  if (y2 == 99)
    #max car can be past pedestrian
    if (y1 == 99)
      py = 7/12
    else
      py = 0.5 * (y1 - 93)/6
    end
  elseif (y2 == -99)
    #max pedestrian can be in front of car
    if (y1 == -99)
      py = 7/12
    else
      py = 0.5 * (-93 - y1)/6
    end
  elseif (y1 == y2)
    #regular transition
    py = 1/6
  else
    py = 1/12
  end

  p = px * py
  return p
end

function hazardPassed(y)
  #println("hazardPassed")
  #checks if you're a safe distance from the pedestrian
  safe = (y >= 99)
  return safe
end

function isCrash(x, y1, y2)
  #println("isCriticalEvent")
  #checks if there was a collision, regardless of speed
  crash = (x==0 && y1 < 0 && y2 >= 0)
  return crash
end

#Main control---------------------------------------------------------------

function MCTS(s, d, pairs)
  #println("MCTS")
  run = 0
  while (run<numRuns)
    pairs = simulate(s,d,pairs)[2]
    run = run + 1
    numPairs = size(pairs, 1)
    for i = 1:numPairs  #short circuits the rest of the checking if you have crash directly available
      if (pairs[i, Symbol("s")] == s && pairs[i, Symbol("r")] == 0)
        run = numRuns
        break
      end
    end
  end
  a = maxQaction(s, pairs)
  return [a, pairs]
end

function simulate(s, d, pairs)
  #println("simulate")
  #check to make sure we're not already done
  if (d == 0)
    return [0, pairs]
  end
  #if the state isn't one we've seen before, add it and go to rollout
  asAndNs = getAsAndNs(s, pairs)
  As = asAndNs[1]
  Ns = asAndNs[2]
  exists = asAndNs[3]
  if (exists == 0)
    pairs = addPair(s, 0, -Inf, -Inf, pairs) #q and r are -inf so this non-existant action isn't chosen. r is 0
    return rollout(s, d, pairs)
  end

  #increment N
  #get new values for As and Ns
  asAndNs = getAsAndNs(s, pairs)
  As = asAndNs[1]
  Ns = asAndNs[2] + 1 #increment Ns until Nsa can be updated

  #check first widening criteria to explore or exploit
  explore = exploreOrExploit(As, Ns)
  if (explore)
    #select a random action and add the s,a pair to the table of pairs
    a = getAction()
    pairs = addPair(s, a, 0, -Inf, pairs) #initialize reward to -Inf and utility to 0
  else
    a = exploit(s, Ns, pairs)
  end

  #check second widening criteria (here just if N(s,a)>0)
  Nsa = getQandN(s, a, pairs)[2]
  if (Nsa == 0) #state action pair has not been found before
    nextStepInfo = nextStep(s, a)
    P = nextStepInfo[1]
    E = nextStepInfo[2]
    sPrime = nextStepInfo[3]
    terminal = nextStepInfo[4]
    r = reward(P, E, terminal)
    #lines 24 - 29  of algorithm in paper since this statement only happens when V(s,a) = 0, the if conditional in line 24 is always true
    pairs = addPair(s, a, 0, r, pairs)
  else
    sPrime = convert(Int64,nextStep(s, a)[3])
    r = tableReward(s, a, pairs)
  end
  if (r == 0) depth = 1 end #if you get to a crash, end this branch of the search so you don't add -inf
  qAndPairs = simulate(sPrime, d-1, pairs)
  #these next steps are where it sends the information back up the ladder it went down
  q = r + gamma*qAndPairs[1]
  pairs = qAndPairs[2]
  pairs = incrementN(s, a, pairs) #Algorithm has this here but that means we get reduncant state action pairs
  pairs = updateQ(s, a, q, pairs)
  return [q, pairs]
end

function rollout(s, depth, pairs)
  #println("rollout")
  #repeatedly samples state transitions until the desired depth (or termination) is reached.
  if (depth == 0)
    return [0, pairs]
  end
  a = getAction()
  nextStepInfo = nextStep(s, a)
  P = nextStepInfo[1]
  E = nextStepInfo[2]
  sPrime = nextStepInfo[3]
  terminal = nextStepInfo[4]
  r = reward(P, E, terminal)
  if (r == 0) depth = 1 end #if you get to a crash, end this branch of the search so you don't add -inf
  rAndPairs = rollout(sPrime, depth - 1, pairs)
  newR = r + gamma * rAndPairs[1]
  pairs = rAndPairs[2]
  return [newR, pairs]
end

function oneDrive(pairs)
  #println("oneDrive")
  #drives through the course once to try to get a crash using MCTS as a guide
  state = initialize()
  depth = 50
  terminal = 0
  r = 0
  while (terminal ==0)
    search = MCTS(state, depth, pairs)
    action = search[1]
    #could keep track of actions here
    pairs = search[2]
    newStep = nextStep(state, action)
    p = newStep[1]
    crash = newStep[2]
    state = newStep[3]
    terminal = newStep[4]
    roundReward = reward(p, crash, terminal)
    r = r + roundReward
  end
  return [r, pairs]
end

function execute(pairs)
  #println("execute")
  #drives a defined # of times and records crashes and their probabilities
  driveRewards = ones(numDrives)
  crashCount = 0
  for i = 1:numDrives
    println(i-1," drives in and ",crashCount, " crashes so far")
    drive = oneDrive(pairs)
    driveRewards[i] = drive[1]
    pairs = drive[2]
    if (driveRewards[i] > -Inf) crashCount = crashCount + 1 end
  end
  println("Number of crashes = ",crashCount)
  pTotal = 0
  for i = 1:100
    if (driveRewards[i] != -Inf) pTotal = pTotal + e^driveRewards[i] end
  end
  println("Total Probability = ", pTotal)
  println("Max Probability = ", e^maximum(driveRewards))

  return [driveRewards, pairs]
end

function basicDrive()
  #println("basicDrive")
  #drives through the course once to try to get a crash using MCTS as a guide
  state = initialize()
  terminal = 0
  r = 0

  while (terminal ==0)
    action = getAction()
    newStep = nextStep(state, action)
    p = newStep[1]
    crash = newStep[2]
    state = newStep[3]
    terminal = newStep[4]
    roundReward = reward(p, crash, terminal)
    r = r + roundReward
  end
  return r
end

function basicExecute()
  #println("basicExecute")
  #drives a defined # of times and records crashes and their probabilities
  driveRewards = ones(numDrives)
  crashCount = 0
  for i = 1:numDrives
    driveRewards[i] = basicDrive()
    if (driveRewards[i] > -Inf) crashCount = crashCount + 1 end
  end
  println("Number of crashes = ",crashCount)
  pTotal = 0
  for i = 1:100
    if (driveRewards[i] != -Inf) pTotal = pTotal + e^driveRewards[i] end
  end
  println("Total Probability = ", pTotal)
  println("Max Probability = ", e^maximum(driveRewards))
  return [driveRewards]
end
