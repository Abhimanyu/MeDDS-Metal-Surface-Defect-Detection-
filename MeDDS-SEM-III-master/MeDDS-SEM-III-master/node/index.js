const express = require('express')
const cors = require('cors')
const app = express()
app.use(express.urlencoded({ extended: true }))
app.use(express.json())
app.use(cors)
app.get('/', async (req, res) => {
    // console.log(req.body)
    const delay = require('delay');
    await delay(1000)
    res.send('okay')
})
app.listen(8080, () => console.log(''))
