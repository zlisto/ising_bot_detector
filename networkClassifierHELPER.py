import math
import networkx as nx
from collections import defaultdict
from operator import itemgetter
import numpy as np
import re
import pandas as pd
from collections import Counter


########################################################################################
#function to pull out source tweet text and source tweeter from a retweet string
def extract_source(text):
    x = re.match(".*RT\W@(\w+):", text)
    if x:
        return x.groups()[0]
    else:
        return x
    
########################################################################################
#function to create rewteet graph from dataframe of tweets 
#edge format is (retweeter,source,num_retweets)
def retweet_network_from_tweets_for_bot_detection(df,cmax=1e8):
   c=0
   e=0
   EdgeList = []
   G = nx.DiGraph()

   for key,tweet in df.iterrows():
       c+=1
       if c> cmax:break
       if c%100000==0:print("\tTweet %s of %s, %s retweets"%(c,len(df),e))
       if 'RT @' in tweet.text:
           try:
               source = extract_source(tweet.text)
               if source:
                   retweeter = tweet.screen_name
                   EdgeList.append((source,retweeter))
                   e +=1
           except:
               pass
               #print('Error with tweet %s: %s'%(tweet['screen_name'],tweet['text']))
   W = Counter(EdgeList)
   ne = len(W)
   print("went through %s tweets, found %s retweets, %s retweet edges"%(c,e,ne))
   
   e = 0
   for edge in W.keys():  
       source = edge[0]
       retweeter = edge[1]
       if not(G.has_node(source)):
           G.add_node(source)
       if not(G.has_node(retweeter)):
           G.add_node(retweeter)
       G.add_edge(retweeter,source, weight = W[edge])
   
   nv = G.number_of_nodes()
   ne = G.number_of_edges()
   print('Retweet network has %s nodes and %s edges'%(nv,ne))
   return G  


#######################################################################################################
#Ising model algorithm
#Gretweet = retweet graph with edge format (retweeter, source, num_retweets)
#PiBot = dictionary of prior bot probabilities for each node
#q_deg = quantile of degree distributions to use for cutoff in interaction energy
#lambda00 = scale of human-human interaction energy
#lambda11 = scale of bot-bot interaction energy
#epsilon  = factor that sets bound on lambda01 (human-bot interaction)
#mu = relative scale of interaction energy to node energy
#PiBotFinal = dictionary of bot probability for each node in Gretweet
def ising_bot_detector(Gretweet,PiBot = None, q_deg=0.999, lambda00=0.6,lambda11=0.8, epsilon = 0.001,mu = 1 ):
    if PiBot==None:
        PiBot = {}
        for v in Gretweet.nodes(): 
            PiBot[v]=0.5
    Din = Gretweet.in_degree(weight='weight')
    Dout = Gretweet.out_degree(weight='weight')
    

    alpha_in =  np.quantile([x[1] for x in Din] ,q_deg)
    alpha_out = np.quantile([x[1] for x in Dout],q_deg)
    alpha=[mu,alpha_out,alpha_in]
    
    print('\tComputing graph cut')
    #link_data[0] = [u,v,is (u,v) in E, is (v,u) in E, number times u rewteets v]
    link_data = getLinkDataRestrained(Gretweet)
    edgelist_data =[(i[0], i[1], psi(i[0],i[1],i[4], Din, Dout,alpha,lambda00,lambda11,epsilon) ) for i in link_data]
    H, BotsIsing, user_data = computeH(Gretweet, PiBot, edgelist_data, Dout, Din)

    print('\tCalculating Ising bot probabilities')

    PiBotFinal = {}
    
    for counter,node in enumerate(Gretweet.nodes()):
        if counter%100000==0:print("Node %s"%counter)
        neighbors=list(np.unique([i for i in nx.all_neighbors(H,node) if i not in [0,1]])) 
        ebots=list(np.unique(np.intersect1d(neighbors,BotsIsing))) 
        ehumans=list(set(neighbors)-set(ebots)) 
        psi_l= sum([H[node][j]['capacity'] for j in ehumans])- sum([H[node][i]['capacity'] for i in ebots]) 

        psi_l_bis= psi_l + H[node][0]['capacity'] - H[1][node]['capacity'] ##probability to be in 1 = notPL

        if (psi_l_bis)>12:
            PiBotFinal[node] = 0
        else:
            PiBotFinal[node] = 1./(1+np.exp(psi_l_bis)) #Probability of being a bot conditioned on labels of neighbors
    return PiBotFinal
    

#####################################################################################################
############################################################################
####################### BUILD/CUT ENERGY GRAPH #############################
############################################################################
'''
Takes as input the RT graph and builds the energy graph.
Then cuts the energy graph to classify
'''
def computeH(G, piBot ,edgelist_data, graph_out, graph_in):
	H=nx.DiGraph()
	'''
	INPUTS:
	## G (ntwkX graph) 
		the Retweet Graph from buildRTGraph
	## piBot (dict of floats)
		a dictionnary with prior on bot probabilities. Keys are users_ids, values are prior bot scores.
	## edgelist_data (list of  tuples) 	
		information about edges to build energy graph. 
		This list comes in part from the getLinkDataRestrained method
	## graph_out (dict of ints)
		a graph that stores out degrees of accounts in retweet graph
	## graph_in (dict of ints)
		a graph that stores in degrees of accounts in retweet graph

	'''
	user_data={i:{
				'user_id':i,
				'out':graph_out[i],
				'in':graph_in[i],
				'old_prob': piBot[i],
				'phi_0': max(0,-np.log(float(10**(-20)+(1-piBot[i])))), 
				'phi_1': max(0,-np.log(float(10**(-20)+ piBot[i]))),
				'prob':0,
				'clustering':0
						} for i in G.nodes()}
	
	set_1 = [(el[0],el[1]) for el in edgelist_data]
	set_2 = [(el[1],el[0]) for el in edgelist_data]
	set_3 = [(el,0) for el in user_data]
	set_4 = [(1,el) for el in user_data]

	H.add_edges_from(set_1+set_2+set_3+set_4,capacity=0)
	

	for i in edgelist_data:
		
		val_00 = i[2][0]
		val_01 = i[2][1]
		val_10 = i[2][2]
		val_11 = i[2][3]

		H[i[0]][i[1]]['capacity']+= 0.5*(val_01+val_10-val_00-val_11)
		H[i[1]][i[0]]['capacity'] += 0.5*(val_01+val_10-val_00-val_11)
		H[i[0]][0]['capacity'] += 0.5*val_11+0.25*(val_10-val_01)
		H[i[1]][0]['capacity'] += 0.5*val_11+0.25*(val_01-val_10)
		H[1][i[0]]['capacity'] += 0.5*val_00+0.25*(val_01-val_10)
		H[1][i[1]]['capacity'] += 0.5*val_00+0.25*(val_10-val_01) 


		if(H[1][i[0]]['capacity']<0):
			print("Neg capacity")
			break;
		if(H[i[1]][0]['capacity']<0):
			print("Neg capacity")
			break;
		if(H[1][i[1]]['capacity']<0):
			print("Neg capacity")
			break;
		if(H[i[0]][0]['capacity']<0):
			print("Neg capacity")
			break;

	for i in user_data.keys():
		H[1][i]['capacity'] += user_data[i]['phi_0']
		if(H[1][i]['capacity'] <0):
			print("Neg capacity");
			break;
			
		H[i][0]['capacity'] += user_data[i]['phi_1']
		if(H[i][0]['capacity'] <0):
			print("Neg capacity");
			break;
	cut_value,mc=nx.minimum_cut(H,1,0)
	PL=list(mc[0]) #the other way around
	if 1 not in PL:
		print("Double check")
		PL=list(mc[1])
	PL.remove(1) 

	return H, PL, user_data

###############################################################################
####################### COMPUTE EDGES INFORMATION #############################
###############################################################################
'''
Takes as input the RT graph and retrieves information on edges 
to further build H.
'''
def getLinkDataRestrained(G):
	'''
	INPUTS:
	## G (ntwkX graph) 
	'''
	edges = G.edges(data=True)
	e_dic = dict(((x,y), z['weight']) for x, y, z in edges)
	link_data = []
	for e in e_dic:
			i=e[0]
			j=e[1]
			rl=False
			wrl=0
			if((j,i) in e_dic.keys()):
				rl = True
				wrl = e_dic[(j,i)]
			link_data.append([i,j,True,rl, e_dic[e], wrl])
	return link_data;




##########################################################################
####################### POTENTIAL FUNCTION ###############################
##########################################################################
'''
Compute joint energy potential between two users
'''
def psi(u1, u2, wlr, in_graph, out_graph,alpha,alambda1,alambda2,epsilon):
	'''
	INPUTS:
	## u1 (int) 
		ID of user u1 
	## u2 (int) 
		ID of user u2
	## wlr (int) 
		number of retweets from u1 to u2
	## out_graph (dict of ints)
		a graph that stores out degrees of accounts in retweet graph
	## in_graph (dict of ints)
		a graph that stores in degrees of accounts in retweet graph
	## alpha (list of floats)
		a list containing hyperparams mu, alpha1, alpha2
	## alambda1 (float)
		value of lambda11
	## alambda2 (float)
		value of lambda00
	## epsilon (int)
		exponent such that delta=10^(-espilon), where lambda01=lambda11+lambda00-1+delta
	'''
	
	#here alpha is a vector of length three, psi decays according to a logistic sigmoid function
	val_00 = 0
	val_01 = 0
	val_10 = 0
	val_11 = 0

	if out_graph[u1]==0 or in_graph[u2]==0:
		print("Relationship problem: "+str(u1)+" --> "+str(u2))

	temp = alpha[1]/float(out_graph[u1])-1 + alpha[2]/float(in_graph[u2])-1 
	if temp <10:
		val_01 =wlr*alpha[0]/(1+np.exp(temp))
	else:
		val_01=0

	val_10 = (alambda2+alambda1-1+epsilon)*val_01
	val_00 = alambda2*val_01
	val_11 = alambda1*val_01

	test2 = 0.5*val_11+0.25*(val_10-val_01)
	test3 = 0.5*val_00+0.25*(val_10-val_01)
	if(min(test2,test3)<0):
		print('PB EDGE NEGATIVE')
		val_00 = val_11 = 0.5*val_01
	
	if(val_00+val_11>val_01+val_10):
		print(u1,u2)
		print('psi01',val_01)
		print('psi11',val_11)
		print('psi00',val_00)
		print('psi10',val_10)
		print("\n")

	values = [val_00,val_01,val_10,val_11]
	return values;





