"""
Module for simulating voltage-gated ion channels using Continuous-Time Markov Networks (CTBN).

This module is adapted from the original CTBN-Voltage-Gated project and refactored
to support modular configuration from JSON files.
"""
import numpy as np
from scipy.integrate import solve_ivp

class CTBNMarkovModel():
    """Simulates a voltage-gated ion channel using a CTBN, configured from a dictionary.

    This model is based on a Hodgkin-Huxley-like structure with multiple closed,
    open, and inactivated states. It calculates state transitions and ionic currents
    in response to voltage clamp protocols.
    """
    def __init__(self, config):
        """Initializes the CTBNMarkovModel from a configuration dictionary.

        Args:
            config (dict): A dictionary containing model parameters, typically loaded
                           from a JSON file.
        """
        # Load parameters from config
        self.model_name = config['model_name']
        self.num_states = config['num_states']
        params = config['parameters']
        for key, value in params.items():
            setattr(self, key, value)

        # Default initializations
        self.NumSwps = 0
        self.demonstrate_cooperative_transition = False
        self.k_coop = 100.0
        self.k_phantom = 1000000.0
        self.A = 0
        self.I = 0
        self.vm = -80  # Default starting voltage

        # Calculate derived parameters
        self.alfac = np.sqrt(np.sqrt((self.ConHiCoeff / self.ConCoeff)))
        self.btfac = np.sqrt(np.sqrt((self.CoffCoeff / self.CoffHiCoeff)))

        # Initialize simulation arrays and starting state
        self.init_waves()
        self.update_rates()
        self.CurrVolt()
        self.state_probs_flat = self.EquilOccup(self.vm)
        self.create_default_protocol()

    def init_waves(self):
        """Initializes data structures for storing pre-calculated rate constants."""
        self.vt = np.arange(-200, 201)
        self.iscft = np.zeros_like(self.vt)
        self.state_probs_flat = np.zeros(self.num_states)
        self.state_probs_flat[0] = 1.0
        num_v = len(self.vt)
        self.fwd_rates_I0 = np.zeros((num_v, 5))
        self.fwd_rates_I1 = np.zeros((num_v, 5))
        self.bwd_rates_I0 = np.zeros((num_v, 5))
        self.bwd_rates_I1 = np.zeros((num_v, 5))
        self.inact_on_rates = np.zeros((num_v, 6))
        self.inact_off_rates = np.zeros((num_v, 6))
        self.update_rates()

    def update_rates(self):
        """Updates all voltage-dependent rate constants."""
        self.stRatesVolt()

    def stRatesVolt(self):
        """Calculates and stores state transition rates as a function of voltage."""
        vt = self.vt
        amt = (self.alcoeff * np.exp((vt / self.alslp)))
        bmt = (self.btcoeff * np.exp(((-vt) / self.btslp)))
        gmt = (self.gmcoeff * np.exp((vt / self.gmslp)))
        dmt = (self.dlcoeff * np.exp(((-vt) / self.dlslp)))
        konlo = (self.ConCoeff * np.exp((vt / self.ConSlp)))
        kofflo = (self.CoffCoeff * np.exp(((-vt) / self.CoffSlp)))
        konop = (self.OpOnCoeff * np.exp((vt / self.OpOnSlp)))
        koffop = (self.OpOffCoeff * np.exp(((-vt) / self.OpOffSlp)))
        for a in range(5):
            self.fwd_rates_I0[:, a] = np.minimum(((4 - a) * amt) if a < 4 else gmt, self.ClipRate)
        for a in range(5):
            self.bwd_rates_I0[:, a] = np.minimum(((a + 1) * bmt) if a < 4 else dmt, self.ClipRate)
        for a in range(5):
            self.fwd_rates_I1[:, a] = np.minimum((((4 - a) * amt) * self.alfac) if a < 4 else gmt, self.ClipRate)
        for a in range(5):
            self.bwd_rates_I1[:, a] = np.minimum((((a + 1) * bmt) / self.btfac) if a < 4 else dmt, self.ClipRate)
        alfac_powers = np.array([self.alfac ** a for a in range(5)])
        btfac_powers = np.array([self.btfac ** a for a in range(5)])
        for a in range(5):
            self.inact_on_rates[:, a] = np.minimum((konlo * alfac_powers[a]), self.ClipRate)
            self.inact_off_rates[:, a] = np.minimum((kofflo / btfac_powers[a]), self.ClipRate)
        self.inact_on_rates[:, 5] = np.minimum(konop, self.ClipRate)
        self.inact_off_rates[:, 5] = np.minimum(koffop, self.ClipRate)

    def CurrVolt(self):
        """Calculates the ionic current at various membrane potentials using GHK equation."""
        scaled_PNasc = self.PNasc
        v_volts = self.vt * 0.001
        near_zero = np.abs(v_volts) < 1e-6
        not_zero = ~near_zero
        self.iscft = np.zeros_like(v_volts)
        if np.any(near_zero):
            du2_zero = (self.F * self.F) / (self.Rgc * self.Tkel)
            self.iscft[near_zero] = scaled_PNasc * du2_zero * (self.Nai - self.Nao)
        if np.any(not_zero):
            v_nz = v_volts[not_zero]
            du1 = (v_nz * self.F) / (self.Rgc * self.Tkel)
            du3 = np.exp(-du1)
            du5_corrected = (self.F * du1 * (self.Nai - self.Nao * du3)) / (1 - du3)
            self.iscft[not_zero] = scaled_PNasc * du5_corrected

    def EquilOccup(self, vm):
        """Calculates the equilibrium (steady-state) occupancies of all states."""
        self.vm = vm
        self.update_rates()
        vidx = np.argmin(np.abs(self.vt - vm))
        fwd_I0, bwd_I0 = self.fwd_rates_I0[vidx], self.bwd_rates_I0[vidx]
        fwd_I1, bwd_I1 = self.fwd_rates_I1[vidx], self.bwd_rates_I1[vidx]
        rel_prob_A_I0 = np.ones(6)
        rel_prob_A_I0[1:] = np.cumprod(fwd_I0 / bwd_I0)
        rel_prob_A_I1 = np.ones(6)
        rel_prob_A_I1[1:] = np.cumprod(fwd_I1 / bwd_I1)
        rel_prob_A_I0 /= rel_prob_A_I0.sum()
        rel_prob_A_I1 /= rel_prob_A_I1.sum()
        inact_on, inact_off = self.inact_on_rates[vidx], self.inact_off_rates[vidx]
        total_rate_I0_to_I1 = np.dot(rel_prob_A_I0, inact_on)
        total_rate_I1_to_I0 = np.dot(rel_prob_A_I1, inact_off)
        rel_prob_I1 = (total_rate_I0_to_I1 / total_rate_I1_to_I0) if total_rate_I1_to_I0 > 0 else 0
        prob_I0 = 1 / (1 + rel_prob_I1)
        prob_I1 = rel_prob_I1 / (1 + rel_prob_I1)
        eq_probs_flat = np.zeros(12)
        eq_probs_flat[:6] = rel_prob_A_I0 * prob_I0
        eq_probs_flat[6:12] = rel_prob_A_I1 * prob_I1
        return eq_probs_flat

    def NowDerivs(self, t, y):
        """Calculates the time derivatives of state probabilities (dy/dt) for the ODE solver."""
        dstdt = np.zeros_like(y)
        vidx = np.searchsorted(self.vt, self.vm)
        vidx = min(max(vidx, 0), len(self.vt) - 1)
        
        current_fwd_I0 = self.fwd_rates_I0[vidx]
        current_bwd_I0 = self.bwd_rates_I0[vidx]
        current_fwd_I1 = self.fwd_rates_I1[vidx]
        current_bwd_I1 = self.bwd_rates_I1[vidx]
        current_inact_on = self.inact_on_rates[vidx]
        current_inact_off = self.inact_off_rates[vidx]

        probs_I0, probs_I1 = y[:6], y[6:12]
        deriv_I0, deriv_I1 = dstdt[:6], dstdt[6:12]

        for i in range(5):
            flux_fwd = current_fwd_I0[i] * probs_I0[i]
            deriv_I0[i] -= flux_fwd
            deriv_I0[i+1] += flux_fwd
            flux_bwd = current_bwd_I0[i] * probs_I0[i+1]
            deriv_I0[i+1] -= flux_bwd
            deriv_I0[i] += flux_bwd

            flux_fwd_i1 = current_fwd_I1[i] * probs_I1[i]
            deriv_I1[i] -= flux_fwd_i1
            deriv_I1[i+1] += flux_fwd_i1
            flux_bwd_i1 = current_bwd_I1[i] * probs_I1[i+1]
            deriv_I1[i+1] -= flux_bwd_i1
            deriv_I1[i] += flux_bwd_i1

        for i in range(6):
            flux_inact = current_inact_on[i] * probs_I0[i]
            deriv_I0[i] -= flux_inact
            deriv_I1[i] += flux_inact
            flux_react = current_inact_off[i] * probs_I1[i]
            deriv_I1[i] -= flux_react
            deriv_I0[i] += flux_react
            
        return dstdt

    def create_default_protocol(self):
        """Creates a default voltage clamp protocol for demonstration."""
        self.BsNm = 'DefaultProtocol'
        self.NumSwps = 1
        self.SwpSeq = np.zeros((12, self.NumSwps))
        sampint = 0.005
        holding_potential = -120
        inactivating_voltage = -10
        test_voltage = -10
        holding_samples = int(200 / sampint)
        inactivating_samples = int(200 / sampint)
        recovery_samples = int(200 / sampint)
        test_samples = int(200 / sampint)
        tail_samples = int(200 / sampint)

        self.SwpSeq[0, :] = 5
        self.SwpSeq[2, :] = holding_potential
        self.SwpSeq[3, :] = holding_samples
        self.SwpSeq[4, :] = inactivating_voltage
        self.SwpSeq[5, :] = holding_samples + inactivating_samples
        self.SwpSeq[6, :] = holding_potential
        self.SwpSeq[7, :] = holding_samples + inactivating_samples + recovery_samples
        self.SwpSeq[8, :] = test_voltage
        self.SwpSeq[9, :] = holding_samples + inactivating_samples + recovery_samples + test_samples
        self.SwpSeq[10, :] = holding_potential
        self.SwpSeq[11, :] = holding_samples + inactivating_samples + recovery_samples + test_samples + tail_samples
        setattr(self, f'SwpSeq{self.BsNm}', self.SwpSeq.copy())
        self.CurrVolt()

    def Sweep(self, SwpNo):
        """Simulates a single sweep of a voltage clamp protocol using an ODE solver."""
        self.vm = self.V_step
        self.update_rates()
        y0 = self.EquilOccup(self.V_hold)
        
        solution = solve_ivp(
            fun=self.NowDerivs,
            t_span=self.t_span,
            y0=y0,
            method='RK45',
            dense_output=True
        )
        return solution

    def generate_current_trace(self, voltage_protocol):
        """Generates a synthetic current trace based on a voltage protocol.

        This method simulates a voltage-clamp experiment and returns the
        resulting current trace, suitable for use as training data.

        Args:
            voltage_protocol (dict): A dictionary defining the voltage steps,
                                     e.g., {'V_hold': -120, 'V_step': 0, 'duration': 0.1}.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The time points of the simulation.
                - np.ndarray: The corresponding current values.
        """
        # Set up the voltage clamp parameters
        self.V_hold = voltage_protocol.get('V_hold', -120.0)
        self.V_step = voltage_protocol.get('V_step', 0.0)
        self.total_duration = voltage_protocol.get('duration', 0.1)
        self.t_span = [0, self.total_duration]

        # Run the simulation sweep
        solution = self.Sweep(SwpNo=0) # SwpNo is a dummy argument here

        # Calculate the current from the solution
        # The open probability (Po) is the sum of probabilities of being in any open state.
        # For the 12-state model, state 5 is the open state.
        # For the 24-state model, states 5 and 17 (5+12) are open states.
        if self.num_states == 12:
            open_states_idx = [5]
        elif self.num_states == 24:
            open_states_idx = [5, 17]
        else:
            open_states_idx = [self.num_states // 2 - 1]

        open_probability = np.sum(solution.y[open_states_idx, :], axis=0)
        current = self.gmax * open_probability * (self.V_step - self.Erev)

        return solution.t, current

class AnticonvulsantCTBNMarkovModel(CTBNMarkovModel):
    """Extends CTBNMarkovModel to include anticonvulsant drug interactions."""
    def __init__(self, config, drug_concentration=0.0, drug_type='DPH'):
        """Initializes the model with drug-specific parameters from a config dict."""
        self.drug_concentration = drug_concentration
        self.drug_type = drug_type.upper()
        
        # Create a temporary config for the superclass constructor
        # by merging base and drug-specific parameters
        temp_config = config.copy()
        temp_config['parameters'] = config['base_parameters'].copy()
        drug_params = config['drug_parameters'][self.drug_type]
        temp_config['parameters'].update(drug_params)

        super().__init__(temp_config)

        # Post-initialization for drug effects
        self._update_drug_rates()
        self.state_probs_flat = self.EquilOccup(self.vm)

    def _update_drug_rates(self):
        """Calculates concentration-dependent on-rates for the drug."""
        self.k_off = self.k_off_base * self.k_off_scaling
        self.KR_resting = self.KI_inactivated * 1000.0
        self.k_on_inactivated_base = self.k_off / self.KI_inactivated
        self.k_on_resting_base = self.k_off / self.KR_resting
        self.k_on_inactivated = self.k_on_inactivated_base * self.drug_concentration
        self.k_on_resting = self.k_on_resting_base * self.drug_concentration

    def EquilOccup(self, vm):
        """Calculates equilibrium occupancies for the 24-state drug model."""
        # This is a placeholder implementation. A full implementation would solve
        # for the null space of the 24x24 transition matrix Q.
        base_eq = super().EquilOccup(vm)
        eq_probs_flat = np.zeros(self.num_states)
        eq_probs_flat[:12] = base_eq * (1 - 0.5) # Assume 50% drug block at equilibrium
        eq_probs_flat[12:] = base_eq * 0.5
        return eq_probs_flat / eq_probs_flat.sum()

    def NowDerivs(self, t, y):
        """Calculates dy/dt for the 24-state model, including drug binding."""
        dstdt = np.zeros(self.num_states)

        # Unpack state probabilities
        # y[:12] are the unbound states
        # y[12:] are the drug-bound states
        unbound_probs = y[:12]
        bound_probs = y[12:]

        # 1. Calculate derivatives for unbound states (same as base model)
        d_unbound = super().NowDerivs(t, unbound_probs)

        # 2. Calculate derivatives for bound states (similar gating, no drug binding)
        # We can reuse the parent's NowDerivs by passing the bound probabilities
        d_bound = super().NowDerivs(t, bound_probs)

        # 3. Calculate fluxes due to drug binding/unbinding
        # For simplicity, assume drug binds to/unbinds from all states at same rate
        # Resting states (first 5)
        for i in range(5):
            binding_flux = self.k_on_resting * unbound_probs[i]
            unbinding_flux = self.k_off * bound_probs[i]
            d_unbound[i] = d_unbound[i] - binding_flux + unbinding_flux
            d_bound[i] = d_bound[i] + binding_flux - unbinding_flux
        
        # Inactivated states (next 6)
        for i in range(6, 12):
            binding_flux = self.k_on_inactivated * unbound_probs[i]
            unbinding_flux = self.k_off * bound_probs[i]
            d_unbound[i] = d_unbound[i] - binding_flux + unbinding_flux
            d_bound[i-6] = d_bound[i-6] + binding_flux - unbinding_flux

        dstdt[:12] = d_unbound
        dstdt[12:] = d_bound
        
        return dstdt