function getBathValue(){
	var uiBathroom = document.getElementsByName("uiBathroom");
	for(var i in uiBathroom) {
		if(uiBathroom[i].checked) {
			return parseInt(i)+1;
		}
	}
	return -1;
}

function getBHKValue() {
	var uiBHK = document.getElementsByName('uiBHK');
	for(var i in uiBHK) {
		if(uiBHK[i].checked) {
			return parseInt(i)+1;
		}
	}
	return -1;
}

function onClickEstimatedPrice() {
	console.log("Estimated Price button clicked");
	var sqft = document.getElementById("uiSqft");
	var bhk = getBHKValue();
	var bathroom = getBathValue();
	var location = document.getElementById("uiLocations");
	var estPrice = document.getElementById("uiEstimatedPrice");
	var url = "http://127.0.0.1:5000/predict_home_price";

	$.post(url, {
		total_sqft: parseFloat(sqft.value),
		bhk: bhk,
		bath: bathroom,
		location: location.value
	}, function(data, status) {
		price_num = parseFloat(data.estimated_price);
		// console.log(data.estimated_price)
		if(price_num>=100){
			price_num = (price_num/100.0).toFixed(2);
			estPrice.innerHTML = "<h2>INR. " + price_num.toString() + " Crore</h2>";
			console.log(status);
		} else {
			estPrice.innerHTML = "<h2>INR. " + price_num.toString() + " Lakh</h2>";
			console.log(status); 
				}
	})
} 

function onPageLoad() {
	console.log("document loaded");
	var url = "http://127.0.0.1:5000/get_location_names";
	$.get(url, function(data, status){
		console.log("got response for get_location_names request");
		if(data) {
			var location = data.locations;
			var uiLocations = document.getElementById("uiLocations");
			$('#uiLocations').empty();
			for(var i in location) {
				var opt = new Option(location[i]);
				$('#uiLocations').append(opt);
			}
		}
	});
}

window.onload = onPageLoad;