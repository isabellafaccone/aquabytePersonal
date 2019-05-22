const _ = require('lodash'),
      async = require('async'),
      request = require('request'),
      fs = require('fs'),
      Json2csvTransform = require('json2csv').Transform;

const fields = [ 'id',
  'localityNo',
  'year',
  'week',
  'hasReportedLice',
  'hasMechanicalRemoval',
  'hasBathTreatment',
  'hasInFeedTreatment',
  'hasCleanerFishDeployed',
  'isFallow',
  'avgAdultFemaleLice',
  'avgMobileLice',
  'avgStationaryLice',
  'seaTemperature',
  'bathTreatments',
  'inFeedTreatments',
  'cleanerFish',
  'mechanicalRemoval',
  'timeSinceLastChitinSynthesisInhibitorTreatment' ];

const outputPath = './out.csv';

const opts = { fields };
const transformOpts = { encoding: 'utf-8' };

//const input = fs.createReadStream(inputPath, { encoding: 'utf8' });
const output = fs.createWriteStream(outputPath, { encoding: 'utf8' });
const json2csv = new Json2csvTransform(opts, transformOpts);
 
const processor = json2csv.pipe(output);

<<<<<<< HEAD
let year = 2016;
=======
let year = 2018;
>>>>>>> 9a56a3ac353998f8f8c6daeaa3518bfd1d6e296f

let month = 1;

let allData = [];

async.whilst(() => { return month <= 52; }, (cb) => {
  let localitiesURL = `https://www.barentswatch.no/api/v1/geodata/fishhealth/locality/${year}/${month}`;

  request({ url: localitiesURL }, (err, resp, body) => {
    let localitiesData = JSON.parse(body);

    console.log(JSON.stringify(localitiesData, null, 2));

    process.exit(0);

    let localityNumbers = [];

    _.forEach(localitiesData.localities, (locality) => {
      if (locality.hasReportedLice) {
        localityNumbers.push(locality.localityNo);
      }
    });

    console.log(`Found ${localityNumbers.length} localities for ${year}/${month}`);

    async.eachLimit(localityNumbers, 5, (localityNumber, cb) => {
      let localityWeekUrl = `https://www.barentswatch.no/api/v1/geodata/fishhealth/locality/${localityNumber}/${year}/${month}`;
    
      request({ url: localityWeekUrl }, (err, resp, body) => {
        let localityWeekData = JSON.parse(body);

        allData.push(localityWeekData.localityWeek);

        json2csv.write(JSON.stringify(localityWeekData.localityWeek));

        debugger;

        cb();
      });
    }, () => {
      console.log('Finished getting data for all localities');

      month++;

      cb();
    });
  });
}, () => {

});