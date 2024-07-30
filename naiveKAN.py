import torch as th
import torch.nn as nn
from itertools import product
from tqdm import tqdm
import plotly.graph_objects as go

class NaiveKANLayer(th.nn.Module):
	def __init__( self, inputdim, outdim, gridsize, *, spline_range=(-2, 2), k=2):
		super(NaiveKANLayer,self).__init__()
		self.gridsize= gridsize
		self.inputdim = inputdim
		self.outdim = outdim
		self.spline_range = spline_range
		self.k = k

		self.coeffs = th.nn.Parameter(th.randn(inputdim, outdim, gridsize))

	def forward(self,x):
		y = th.zeros(*x.shape[:-1], self.outdim)
		for i, j in product(range(self.inputdim), range(self.outdim)):
			xx = x[..., i]
			yy = self._sample_spline_1d(i, j, xx)
			y[..., j] += yy
		return y

	def _b_spline_basis(self, t, k, i, knots):
		if k == 0:
			return th.where((knots[i] <= t) & (t < knots[i + 1]), 1.0, 0.0)

		denom1 = knots[i + k] - knots[i]
		denom2 = knots[i + k + 1] - knots[i + 1]
		
		term1 = ((t - knots[i]) / denom1) * self._b_spline_basis(t, k - 1, i, knots) if denom1 != 0 else 0.0
		term2 = ((knots[i + k + 1] - t) / denom2) * self._b_spline_basis(t, k - 1, i + 1, knots) if denom2 != 0 else 0.0
		
		return term1 + term2

	def _sample_spline_1d(self, i, j, x):
		# Define non-uniformly spaced x-values and corresponding y-values for control points
		x_control = th.linspace(*self.spline_range, self.gridsize, dtype=th.float32)
		y_control = self.coeffs[i, j, :]

		# Number of control points
		num_control_points = len(x_control)
		# Knot vector (adjusted to match the x-values range)
		knot_vector = th.cat((
			th.full((self.k,), x_control[0]),
			x_control,
			th.full((self.k,), x_control[-1])
		))

		y = th.zeros_like(x)
		for i in range(num_control_points):
			y += y_control[i] * self._b_spline_basis(x, self.k, i, knot_vector)
		return y

	def __repr__(self):
		return 'NaiveKANLayer(' \
			+ 'inputdim=' + str(self.inputdim) \
			+ ', outdim=' + str(self.outdim) \
			+ ', gridsize=' + str(self.gridsize) \
			+ ', n_params=' + str(sum(p.numel() for p in self.parameters() if p.requires_grad)) + ')'


if __name__ == '__main__':
	# to_predict = lambda x, y: np.exp(np.sin(x) + np.cos(y))
	to_predict = lambda x: th.exp(th.sin(th.pi*x[:,[0]]) + x[:,[1]]**2)
	x_train = th.rand((1000, 2)) * 4 - 2
	y_train = to_predict(x_train)

	model = nn.Sequential(
		NaiveKANLayer(2, 10, 5, spline_range=(-3, 3), k=3),
		NaiveKANLayer(10, 1, 5, spline_range=(-3, 3), k=3)
	)

	optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
	criterion = nn.MSELoss()

	p_bar = tqdm(range(10000))
	for _ in p_bar:
		optimizer.zero_grad()
		y_pred = model(x_train)
		loss = criterion(y_pred, y_train)
		loss.backward()
		optimizer.step()
		p_bar.set_description(f'Loss: {loss.item():2f}')

	print('Final loss:', loss.item())
	print('Prediction:', y_pred[0].item())
	print('Ground truth:', y_train[0].item())

	fig = go.Figure()
	fig.add_trace(go.Scatter3d(
		x=x_train[:,0].numpy(),
		y=x_train[:,1].numpy(),
		z=y_train[:,0].numpy(),
		mode='markers',
		marker=dict(size=2),
		name='Ground truth'
	))
	fig.add_trace(go.Scatter3d(
		x=x_train[:,0].numpy(),
		y=x_train[:,1].numpy(),
		z=y_pred[:,0].detach().numpy(),
		mode='markers',
		marker=dict(size=2),
		name='Prediction'
	))
	fig.show()
	fig.write_html('naiveKAN.html')
