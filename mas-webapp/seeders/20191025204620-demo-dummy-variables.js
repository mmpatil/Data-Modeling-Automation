'use strict';

module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.bulkInsert('DummyVariable', [{
      RunId: 1,
      Name: "random name",
      Coefficient: 1,
      Pval: 0.2
    }, {
      RunId: 1,
      Name: "random name 2",
      Coefficient: 2.0,
      Pval: 0.4
    },{
      RunId: 2,
      Name: "random name",
      Coefficient: 3.0,
      Pval: 0.2
    },{
      RunId: 3,
      Name: "random name",
      Coefficient: 1.0,
      Pval: 0.3
    }, {
      RunId: 4,
      Name: "random name",
      Coefficient: 1,
      Pval: 0.2
    }, {
      RunId: 4,
      Name: "random name 2",
      Coefficient: 2.0,
      Pval: 0.4
    }], {});
  },

  down: (queryInterface, Sequelize) => {
    return queryInterface.bulkDelete('DummyVariable', null, {});
  }
};
