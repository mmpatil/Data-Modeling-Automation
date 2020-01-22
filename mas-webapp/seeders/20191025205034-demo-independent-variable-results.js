'use strict';

module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.bulkInsert('IndependentVariableResult', [{
        ModelId: 1,
        Name: "IV1",
        Coefficient: 1.0,
        Pval: 0.7,
        Transformations: "???",
        UnitRoot: "I have no idea what this is"
      }, {
        ModelId: 1,
        Name: "IV2",
        Coefficient: 4.0,
        Pval: 0.2,
        Transformations: "???",
        UnitRoot: "I have no idea what this is"
      }, {
        ModelId: 1,
        Name: "IV3",
        Coefficient: 3.2,
        Pval: 0.3,
        Transformations: "???",
        UnitRoot: "I have no idea what this is"
      },
      {
        ModelId: 2,
        Name: "IV1",
        Coefficient: 1.0,
        Pval: 0.7,
        Transformations: "???",
        UnitRoot: "I have no idea what this is"
      }, {
        ModelId: 2,
        Name: "IV2",
        Coefficient: 4.0,
        Pval: 0.2,
        Transformations: "???",
        UnitRoot: "I have no idea what this is"
      }, {
        ModelId: 2,
        Name: "IV3",
        Coefficient: 3.2,
        Pval: 0.3,
        Transformations: "???",
        UnitRoot: "I have no idea what this is"
      }], {});
  },

  down: (queryInterface, Sequelize) => {
    return queryInterface.bulkDelete('IndependentVariableResult', null, {});
  }
};
