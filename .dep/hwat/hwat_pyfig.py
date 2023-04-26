
from pyfig import Pyfig
from pyfig.plugins import Paths

class hwatPyfig(Pyfig):
	
	paths: Paths
	system_name: str		= ''

	@property
	def system_id(self):
		return '_'.join([f'{a_z_i}_{a_i[0]}_{a_i[1]}_{a_i[2]}' for a_z_i, a_i in zip(self.a_z, self.a)])

	n_b: int = 256
	
	@property
	def system_id_path(self):
		return self.paths.dump_dir / f'{self.system_id}.txt'

	charge:     int         = 0
	spin:       int         = 0
	a: NDArray[float, pnd.float32] = Field(default_factory=lambda: np.array([[0.0, 0.0, 0.0],]))
	a_z: NDArray[int, pnd.int32] = Field(default_factory=lambda: np.array([4,]))
	mo_coef: NDArray[float, pnd.float32] = Field(default= None)


	n_corr:     int         = 20
	acc_target: int         = 0.5
	init_data_scale: float  = 1.

	mo_coef: np.ndarray = Field(default= None)

	@property
	def n_e(self):
		return int(sum(self.a_z))

	@property
	def n_u(self):
		return (self.spin + self.n_e) // 2

	@property
	def n_d(self):
		return self.n_e - self.n_u

	@property
	def n_equil_step(self):
		return 10000 // self.n_corr

	loss: str        = ''  # orb_mse, vmc
	compute_energy: bool = False  # true by default

	class Config:
		arbitrary_types_allowed = True

	def init_app(ii):
		""" 
		- want this to happen after pyfig is updated
		"""
		print('\npyfig:pyscf: ')

		ii.mol: gto.Mole = gto.Mole(
			atom	= ii.system_id, 
			basis	='sto-3g', 
			charge	= ii.charge, 
			spin 	= ii.spin, 
			unit	= 'bohr'
		)
		ii.mol.build()
		mean_field_obj = scf.UHF(ii.mol)
		mean_field_obj.kernel()
		ii._hf = mean_field_obj

		# Molecular orbital (MO) coefficients 
		# matrix where rows are atomic orbitals (AO) and columns are MOs
		ii.mo_coef = np.array(mean_field_obj.mo_coeff)
		print('app:init_app: mo_coef shape:', ii.mo_coef.shape)
		mean_field_obj.analyze()
		pprint(mean_field_obj)

	def record_summary(ii, summary: dict= None, opt_obj_all: list= None) -> None:
		import wandb
		summary = summary or {}

		atomic_id = "-".join([str(int(float(i))) for i in ii.a_z.flatten()])
		spin_and_charge = f'{ii.charge}_{ii.spin}'
		geometric_hash = f'{ii.a.mean():.0f}'
		exp_metaid = '_'.join([atomic_id, spin_and_charge, geometric_hash])
		
		if len(opt_obj_all)==0:
			print('no opt_obj_all, setting to 0.0')
			opt_obj_all = np.array([0.0, 0.0])
		
		columns = ["charge_spin_az0-az1-...pmu", "opt_obj", "Error (+/- std)"]
		data = [exp_metaid, np.array(opt_obj_all).mean(), np.array(opt_obj_all).std()]
		
		data += list(summary.values())
		columns += list(summary.keys())

		print('pyfig:app:record_summary:Result ')
		for i, j in zip(columns, data):
			print(i, j)
		print(summary)
		# print(pd.DataFrame.from_dict(summary | dict(zip(columns, data))).to_markdown())

		Result = wandb.Table(columns= columns)
		Result.add_data(*data)
		wandb.log(dict(Result= Result) | (summary or {}))

		return True
	
	def post_init_update(ii):
		systems = {}
		system = systems.get(ii.system_name, {})
		if ii.system_name and not system:
			print('pyfig:app:post_init_update: system not found')
			return system
		
		def ang2bohr(tensor):
			return np.array(tensor) * 1.889725989
		
		unit = system.get('unit', None)
		if unit is None:
			print('pyfig:app:post_init_update: unit not specified, assuming bohr')
			unit = 'bohr'

		if system and 'a' in system and unit.lower() == 'angstrom': 
			system['a'] = ang2bohr(system['a'])
		return system
