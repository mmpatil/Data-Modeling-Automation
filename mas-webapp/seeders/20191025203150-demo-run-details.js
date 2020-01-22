'use strict';

module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.bulkInsert('RunDetail', [{
      StartDate: new Date(),
      EndDate: new Date(),
      Status: "SUCCESS"
    }, {
      StartDate: new Date()
    }, {
      StartDate: new Date(),
      EndDate: new Date(),
      Status: "FAIL"
    }, {
      StartDate: new Date(),
      EndDate: new Date(),
      Status: "SUCCESS"
    }], {});
  },

  down: (queryInterface, Sequelize) => {
    return queryInterface.bulkDelete('RunDetail', null, {});
  }
};
