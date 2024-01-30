const form = document.getElementById('predictForm');

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const input = document.getElementById('text').value;

  try {
    const response = await fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',  // Explicitly set the method to POST
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ news: input }),
    });

    if (response.ok) {
      const prediction = (await response.json()).prediction;
      const resultElement = document.getElementById('result');
      resultElement.innerText = `Prediction: ${prediction}`;
      resultElement.classList.remove('text-red-500', 'text-green-500'); // Remove previous classes
      resultElement.classList.add(prediction === 'FAKE News' ? 'text-red-500' : 'text-green-500');
    } else {
      console.error('Request failed:', response.status);
    }
  } catch (error) {
    console.error('Request failed:', error);
  }
});
