<script>
    let myChart = document.getElementById('myChart').getContext('2d')
    let massPopChart = new Chart(myChart, {
    type:'bar', // bar, horizontalBar, pie, line, doughnut, radar, polarArea
    data:{
    labels:["Boston", ],
    datasets:[]},
    options:{}});
</script>