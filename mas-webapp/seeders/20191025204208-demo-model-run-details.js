'use strict';

module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.bulkInsert('ModelRunDetail', [{
      RunId: 1
    },
    {
      RunId: 1
    },
    {
      RunId: 1
    },
    {
      RunId: 4
    },
    {
      RunId: 4
    },
    {
      RunId: 4
    }], {});
  },

  down: (queryInterface, Sequelize) => {
    return queryInterface.bulkDelete('ModelRunDetail', null, {});
  }
};
