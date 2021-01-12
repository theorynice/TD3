import jsbsim
import random
import numpy as np
import pdb
class jsbsim_env:
	class state():
		def __init__(self):
			self.p=0
			self.q=0
			self.r=0
			self.u=0
			self.v=0
			self.w=0
			self.theta=0
			self.phi=0
			self.psi=0
			self.x=0
			self.y=0
			self.z=0
		def make_ref(self,theta,phi,psi,p,q,r,z):
			self.theta=theta
			self.phi=phi
			self.psi=psi
			self.p=p
			self.q=q
			self.r=r
			self.z=z

		def __sub__(self,other):
			list=[]
			list.append(self.theta-other.theta)
			list.append(self.phi-other.phi)
			list.append(self.psi-other.psi)
			list.append(self.q-other.q)
			list.append(self.p-other.p)
			list.append(self.r-other.r)
			list.append(self.z-other.z)
			return list
		
	def __init__(self, aircraft,delt):
		self.fdm=jsbsim.FGFDMExec('.')
		self.fdm.set_debug_level(0)
		self.fdm.load_model(aircraft)
		self.fdm.set_dt(delt)
		self.fdm.set_property_value("ic/lat-gc-deg",47.0) 
		self.fdm.set_property_value("ic/long-gc-deg",122.0)
		self.fdm.set_property_value("ic/h-sl-ft",2000)
		self.fdm.set_property_value("ic/u-fps",450)
		self.fdm.set_property_value("ic/v-fps",0)
		self.fdm.set_property_value("ic/w-fps",0)
		self.fdm.set_property_value("ic/p-rad_sec",0)
		self.fdm.set_property_value("ic/q-rad_sec",0)
		self.fdm.set_property_value("ic/r-rad_sec",0)
		self.fdm.set_property_value("ic/phi-rad",0)
		self.fdm.set_property_value("ic/theta-rad",1)
		self.fdm.set_property_value("ic/psi-true-rad",0)
		self.fdm.set_property_value('propulsion/set-running', -1)
		self.fdm.run_ic()
		self.ref_state=self.state()
		self.next_state=self.state()
	
	def set_episode_initial(self):
		v=random.sample(range(450,700),1)
		z=random.sample(range(2500,7000),1)
		self.fdm.set_property_value("ic/h-sl-ft",z[0])
		self.fdm.set_property_value("ic/u-fps",v[0])
		self.fdm.set_property_value('propulsion/set-running', -1)
		self.ref_state.make_ref(0,0,self.fdm["ic/psi-true-rad"],0,0,0,self.fdm["ic/h-sl-ft"])
	
	def reset(self):
		self.set_episode_initial()
		self.fdm.run_ic()
		self.fdm.set_sim_time(0)
		reset_state=[]
		reset_state.append(self.fdm["velocities/p-rad_sec"])
		reset_state.append(self.fdm["velocities/q-rad_sec"])
		reset_state.append(self.fdm["velocities/r-rad_sec"])
		reset_state.append(self.fdm["velocities/u-aero-fps"])
		reset_state.append(self.fdm["velocities/v-aero-fps"])
		reset_state.append(self.fdm["velocities/w-aero-fps"])
		reset_state.append(self.fdm["attitude/phi-rad"])
		reset_state.append(self.fdm["attitude/theta-rad"])
		reset_state.append(self.fdm["attitude/psi-rad"])
		reset_state.append(self.fdm["position/eci-x-ft"])
		reset_state.append(self.fdm["position/eci-y-ft"])
		reset_state.append(self.fdm["position/h-sl-ft"])
		return reset_state
	
	def reward_caculator(self):
		done=False;
		if(self.fdm.get_sim_time()>20):
			done=True
		deviate=np.array(self.next_state-self.ref_state)
		#pdb.set_trace()
		reward=-((deviate**2).sum())
		if deviate.sum()<20:
			reward+=1
		return reward,done
	
	def step(self,action):
		self.fdm["fcs/aileron-cmd-norm"]=action[0]
		self.fdm["fcs/elevator-cmd-norm"]=action[1]
		self.fdm["fcs/rudder-cmd-norm"]=action[2]
		self.fdm["fcs/throttle-cmd-norm"]=action[3]
		self.fdm.run()
		next_state=[]
		self.next_state.p=self.fdm["velocities/p-rad_sec"]
		self.next_state.q=self.fdm["velocities/q-rad_sec"]
		self.next_state.r=self.fdm["velocities/r-rad_sec"]
		self.next_state.u=self.fdm["velocities/u-aero-fps"]
		self.next_state.v=self.fdm["velocities/v-aero-fps"]
		self.next_state.w=self.fdm["velocities/w-aero-fps"]
		self.next_state.phi=self.fdm["attitude/phi-rad"]
		self.next_state.theta=self.fdm["attitude/theta-rad"]
		self.next_state.psi=self.fdm["attitude/psi-rad"]
		self.next_state.x=self.fdm["position/eci-x-ft"]
		self.next_state.y=self.fdm["position/eci-y-ft"]
		self.next_state.z=self.fdm["position/h-sl-ft"]
		d=self.next_state.__dict__
		for key in d :
			if '_' not in key:
				next_state.append(d[key])
		step_reward,done=self.reward_caculator()
		return next_state,step_reward,done

	def action_sample(self):
		action=[]
		action.append(random.uniform(0, 1))
		action.append(random.uniform(0, 1))
		action.append(random.uniform(0, 1))
		n=0
		for i in action:
			action[n]=2*(0.5-i)
			n=n+1
		action.append(random.uniform(0, 1))
		#pdb.set_trace()
		return action